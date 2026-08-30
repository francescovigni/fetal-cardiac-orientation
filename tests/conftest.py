"""Synthetic FOCUS-shaped fixtures.

CI must not depend on a 58 MB download, and a test that only runs when the real
dataset happens to be present is not a test.  These fixtures build a miniature
dataset with the same directory layout, annotation formats and conventions as
FOCUS, from ellipses whose parameters are known exactly — so the parsing code,
the crop geometry and the estimators are all exercised against ground truth that
was constructed rather than annotated.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SPLITS = {"train": "training", "val": "validation", "test": "testing"}

# (cx, cy, a, b, theta_deg) — deliberately spans the wrap and a near-round case
ELLIPSES = [
    (240.0, 180.0, 80.0, 55.0, 30.0),
    (250.0, 190.0, 90.0, 62.0, 135.0),
    (230.0, 175.0, 70.0, 68.0, 5.0),
    (245.0, 185.0, 85.0, 40.0, 172.0),
]


def _corners(cx, cy, a, b, theta_deg):
    t = math.radians(theta_deg)
    u = np.array([math.cos(t), math.sin(t)]) * a
    v = np.array([-math.sin(t), math.cos(t)]) * b
    c = np.array([cx, cy])
    return np.stack([c + u + v, c + u - v, c - u - v, c - u + v])


def _render(shape, cx, cy, a, b, theta_deg, filled=True):
    import cv2

    img = np.zeros(shape, np.uint8)
    cv2.ellipse(
        img,
        (int(round(cx)), int(round(cy))),
        (int(round(a)), int(round(b))),
        theta_deg,
        0,
        360,
        255,
        -1 if filled else 2,
    )
    return img


@pytest.fixture(scope="session")
def fake_focus(tmp_path_factory) -> Path:
    """A miniature dataset with FOCUS's layout and annotation conventions."""
    import cv2

    root = tmp_path_factory.mktemp("FOCUS")
    shape = (360, 480)
    for _split, d in SPLITS.items():
        (root / d / "images").mkdir(parents=True)
        (root / d / "annfiles_ellipse").mkdir(parents=True)
        (root / d / "annfiles_rectangle").mkdir(parents=True)
        (root / d / "annfiles_mask").mkdir(parents=True)
        for i, (cx, cy, a, b, th) in enumerate(ELLIPSES, start=1):
            stem = f"{i:03d}"
            # a textured "ultrasound" with a bright elliptical structure in it
            rng = np.random.default_rng(i)
            img = (rng.random(shape) * 60).astype(np.uint8)
            img = np.maximum(img, _render(shape, cx, cy, a, b, th) // 2)
            img = np.maximum(img, _render(shape, cx, cy, a, b, th, filled=False))
            cv2.imwrite(str(root / d / "images" / f"{stem}.png"), img)

            tx, ty, ta, _tb = cx + 10, cy + 5, a * 2.1, b * 2.1  # thorax: near-round
            (root / d / "annfiles_ellipse" / f"{stem}.txt").write_text(
                f"{cx} {cy} {a} {b} {th} cardiac\n{tx} {ty} {ta} {ta} 0.0 thorax\n"
            )
            cc = _corners(cx, cy, a, b, th).ravel()
            tc = _corners(tx, ty, ta, ta, 0.0).ravel()
            (root / d / "annfiles_rectangle" / f"{stem}.txt").write_text(
                " ".join(f"{v:.4f}" for v in cc)
                + " cardiac 0\n"
                + " ".join(f"{v:.4f}" for v in tc)
                + " thorax 0\n"
            )
            cv2.imwrite(
                str(root / d / "annfiles_mask" / f"{stem}-cardiac.png"),
                _render(shape, cx, cy, a, b, th),
            )
            cv2.imwrite(
                str(root / d / "annfiles_mask" / f"{stem}-thorax.png"),
                _render(shape, tx, ty, ta, ta, 0.0),
            )
    return root


@pytest.fixture(scope="session")
def ellipse_mask():
    """A clean filled ellipse and its true axial angle."""

    def make(a=80.0, b=30.0, theta=37.0, shape=(300, 400)):
        return _render(shape, shape[1] / 2, shape[0] / 2, a, b, theta), theta

    return make
