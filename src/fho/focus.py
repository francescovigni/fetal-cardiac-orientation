"""Parsing of the FOCUS dataset (Zenodo 14597550, CC-BY-4.0).

Layout, per split directory (training / validation / testing):

    images/NNN.png                     grayscale, ~961x663
    annfiles_ellipse/NNN.txt           "cx cy a b theta_deg label"   (a = semi-major)
    annfiles_rectangle/NNN.txt         "x1 y1 x2 y2 x3 y3 x4 y4 label difficulty"  (DOTA-style OBB)
    annfiles_mask/NNN-{cardiac,thorax}.png

Labels are ``cardiac`` and ``thorax``.

The ellipse and rectangle annotations are consistent with each other: the OBB long
half-edge equals the ellipse semi-major axis, the OBB centre equals the ellipse
centre, and the long-edge direction equals ``theta``.  This was verified on the
whole dataset by :func:`verify_consistency` — it is what licenses us to use the
ellipse ``theta`` as exact orientation ground truth.

All angles here are in **image coordinates**, i.e. x right, y *down*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLITS = {"train": "training", "val": "validation", "test": "testing"}
LABELS = ("cardiac", "thorax")


@dataclass(frozen=True)
class Ellipse:
    cx: float
    cy: float
    a: float  # semi-major
    b: float  # semi-minor
    theta: float  # degrees, direction of the major axis, image coords
    label: str

    @property
    def center(self) -> np.ndarray:
        return np.array([self.cx, self.cy], float)

    @property
    def anisotropy(self) -> float:
        """b/a in [0, 1].  -> 1 means the orientation is ill-defined."""
        return float(self.b / self.a) if self.a > 0 else 1.0

    def axis_endpoints(self) -> np.ndarray:
        """The four axis endpoints: [major+, major-, minor+, minor-] as (4, 2)."""
        t = math.radians(self.theta)
        u = np.array([math.cos(t), math.sin(t)])  # major direction
        v = np.array([-math.sin(t), math.cos(t)])  # minor direction
        c = self.center
        return np.stack([c + self.a * u, c - self.a * u, c + self.b * v, c - self.b * v])

    def aabb(self) -> tuple[float, float, float, float]:
        """Axis-aligned box (x0, y0, x1, y1) tightly enclosing the ellipse."""
        t = math.radians(self.theta)
        dx = math.hypot(self.a * math.cos(t), self.b * math.sin(t))
        dy = math.hypot(self.a * math.sin(t), self.b * math.cos(t))
        return self.cx - dx, self.cy - dy, self.cx + dx, self.cy + dy


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: Path
    ellipses: dict[str, Ellipse]
    obbs: dict[str, np.ndarray]  # label -> (4, 2) corners
    mask_paths: dict[str, Path]

    @property
    def cardiac(self) -> Ellipse:
        return self.ellipses["cardiac"]

    @property
    def thorax(self) -> Ellipse | None:
        return self.ellipses.get("thorax")


def _read_ellipse_file(p: Path) -> dict[str, Ellipse]:
    out: dict[str, Ellipse] = {}
    for line in p.read_text().strip().splitlines():
        f = line.split()
        if len(f) < 6:
            continue
        cx, cy, a, b, th = (float(x) for x in f[:5])
        if b > a:  # normalise: a is always the semi-major
            a, b, th = b, a, th + 90.0
        out[f[5]] = Ellipse(cx, cy, a, b, th % 180.0, f[5])
    return out


def _read_obb_file(p: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for line in p.read_text().strip().splitlines():
        f = line.split()
        if len(f) < 9:
            continue
        out[f[8]] = np.array([float(x) for x in f[:8]], float).reshape(4, 2)
    return out


def load_split(root: str | Path, split: str) -> list[Sample]:
    """Load one split.  ``split`` is one of train / val / test."""
    root = Path(root)
    d = root / SPLITS[split]
    samples = []
    for img in sorted((d / "images").glob("*.png")):
        stem = img.stem
        ell = _read_ellipse_file(d / "annfiles_ellipse" / f"{stem}.txt")
        obb = _read_obb_file(d / "annfiles_rectangle" / f"{stem}.txt")
        masks = {
            lab: d / "annfiles_mask" / f"{stem}-{lab}.png"
            for lab in LABELS
            if (d / "annfiles_mask" / f"{stem}-{lab}.png").exists()
        }
        if "cardiac" not in ell:
            continue
        samples.append(Sample(stem, img, ell, obb, masks))
    return samples


def obb_angle(corners: np.ndarray) -> float:
    """Long-edge direction of a 4-corner OBB, degrees mod 180."""
    corners[1:] - corners[0]
    lens = np.linalg.norm(np.stack([corners[1] - corners[0], corners[2] - corners[1]]), axis=1)
    edge = (corners[1] - corners[0]) if lens[0] >= lens[1] else (corners[2] - corners[1])
    return math.degrees(math.atan2(edge[1], edge[0])) % 180.0


def verify_consistency(samples: list[Sample], label: str = "cardiac") -> dict:
    """Cross-check ellipse annotations against the OBB annotations.

    Returns max/median discrepancy in centre (px), semi-axis (px) and angle (deg).
    A clean result is the evidence that ``Ellipse.theta`` is trustworthy ground truth.
    """
    from .geometry import angdiff_axial

    dc, da, dth = [], [], []
    for s in samples:
        if label not in s.ellipses or label not in s.obbs:
            continue
        e, c = s.ellipses[label], s.obbs[label]
        dc.append(np.linalg.norm(e.center - c.mean(0)))
        lens = np.linalg.norm(np.stack([c[1] - c[0], c[2] - c[1]]), axis=1) / 2
        da.append(abs(max(lens) - e.a))
        dth.append(angdiff_axial(obb_angle(c), e.theta))

    def f(v):
        return dict(
            median=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(np.max(v))
        )

    return {"n": len(dc), "center_px": f(dc), "semi_major_px": f(da), "angle_deg": f(dth)}
