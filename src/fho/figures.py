"""Figures for the README and the article.

    python -m fho.figures --out docs/figures

Everything is generated from the trained checkpoints, so a figure can never drift
away from the numbers it illustrates.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import geometry as G
from .focus import load_split
from .landmarks import CropSpec, make_example

GT, PRED = "#2a9d8f", "#e76f51"
SERIES = ["#264653", "#e76f51", "#e9c46a", "#8ab17d"]
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def _load(ckpt: Path):
    import torch
    from .landmarks import K
    from .model import LandmarkNet

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = LandmarkNet(K, state.get("width", 32))
    model.load_state_dict(state["model"])
    model.eval()
    return model, CropSpec(**state["spec"])


def _predict_all(model, spec, samples):
    import torch

    rows = []
    for s in samples:
        ex = make_example(s, spec)
        with torch.no_grad():
            axis, coords = model(torch.from_numpy(ex["image"])[None, None])
        pts = coords[0].numpy() * 4.0
        d = G.axis_from_landmarks(pts)
        rows.append(dict(
            image=ex["image"], pts=pts, stem=s.stem,
            gt=ex["gt_angle"], pred=d["angle"],
            direct=float(G.decode_axial(axis[0].numpy())),
            err=float(G.angdiff_axial(d["angle"], ex["gt_angle"])),
            aniso=ex["anisotropy"], size=s.cardiac.a,
            gt_pts=ex["landmarks"],
        ))
    return rows


def _draw_axis(ax, center, angle_deg, length, color, label, ls="-"):
    t = math.radians(angle_deg)
    u = np.array([math.cos(t), math.sin(t)]) * length / 2
    ax.plot([center[0] - u[0], center[0] + u[0]],
            [center[1] - u[1], center[1] + u[1]],
            color=color, lw=1.8, ls=ls, label=label, solid_capstyle="round")


def fig_qualitative(rows, out: Path, n=6):
    """Best, median and worst cases — never only the flattering ones."""
    order = np.argsort([r["err"] for r in rows])
    pick = list(order[:2]) + list(order[len(order) // 2 - 1:len(order) // 2 + 1]) + list(order[-2:])
    tags = ["best", "best", "median", "median", "worst", "worst"]

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.2))
    for ax, i, tag in zip(axes.ravel(), pick[:n], tags):
        r = rows[i]
        ax.imshow(r["image"], cmap="gray", vmin=0, vmax=1)
        c = r["pts"][:2].mean(0)
        L = np.linalg.norm(r["pts"][0] - r["pts"][1])
        _draw_axis(ax, r["gt_pts"][:2].mean(0), r["gt"],
                   np.linalg.norm(r["gt_pts"][0] - r["gt_pts"][1]), GT, "ground truth")
        _draw_axis(ax, c, r["pred"], L, PRED, "predicted", ls="--")
        ax.scatter(r["pts"][:2, 0], r["pts"][:2, 1], s=20, c=PRED, zorder=3, lw=0)
        ax.scatter(r["pts"][2:, 0], r["pts"][2:, 1], s=18, facecolors="none",
                   edgecolors=PRED, zorder=3, lw=1.0)
        ax.set_title(f"{tag} · {r['stem']} · {r['err']:.1f}°", fontsize=7.5)
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    axes[0, 0].legend(loc="lower left", fontsize=6, framealpha=0.85)
    fig.suptitle("Predicted heart long axis vs annotation — best, median and worst test cases",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "qualitative.png", bbox_inches="tight")
    plt.close(fig)


def fig_agreement(rows, out: Path):
    """Bland-Altman plus the error distribution — agreement, not accuracy."""
    gt = np.array([r["gt"] for r in rows])
    pr = np.array([r["pred"] for r in rows])
    d = G.signed_angdiff_axial(pr, gt)
    mean = G.circ_mean_axial  # noqa: F841  (kept for clarity of intent)
    bias, sd = d.mean(), d.std(ddof=1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 3.0))
    a1.axhline(bias, color=PRED, lw=1.2, label=f"bias {bias:+.2f}°")
    for k, ls in ((1.96, "--"),):
        a1.axhline(bias + k * sd, color=PRED, lw=0.9, ls=ls,
                   label=f"95% LoA ±{k*sd:.1f}°")
        a1.axhline(bias - k * sd, color=PRED, lw=0.9, ls=ls)
    a1.scatter(gt, d, s=16, c="#264653", alpha=0.75, lw=0)
    a1.set_xlabel("annotated angle (deg)")
    a1.set_ylabel("predicted − annotated (deg)")
    a1.set_title("Bland–Altman", fontsize=9)
    a1.legend(fontsize=6.5, loc="upper right")

    err = np.abs(d)
    a2.hist(err, bins=np.arange(0, err.max() + 2.5, 2.5), color="#264653", alpha=0.85)
    a2.axvline(np.median(err), color=PRED, lw=1.4, label=f"median {np.median(err):.2f}°")
    a2.axvline(np.percentile(err, 90), color=PRED, lw=1.0, ls="--",
               label=f"p90 {np.percentile(err, 90):.2f}°")
    a2.set_xlabel("absolute error (deg)")
    a2.set_ylabel("test images")
    a2.set_title("Error distribution", fontsize=9)
    a2.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "agreement.png", bbox_inches="tight")
    plt.close(fig)


def fig_risk_coverage(rows, out: Path):
    """Error against retained fraction, for each candidate confidence signal."""
    err = np.array([r["err"] for r in rows])
    signals = {
        "head agreement": -np.array([G.angdiff_axial(r["pred"], r["direct"]) for r in rows]),
        "shape elongation": -np.array([r["aniso"] for r in rows]),
        "heart size": np.array([r["size"] for r in rows]),
        "oracle": -err,
    }
    cov = np.linspace(1.0, 0.3, 15)
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for (name, s), col in zip(signals.items(), SERIES):
        e = err[np.argsort(-s)]
        med = [np.median(e[:max(int(round(c * len(e))), 1)]) for c in cov]
        ax.plot(cov * 100, med, color="#9c6644" if name == "oracle" else col,
                lw=1.0 if name == "oracle" else 1.6,
                ls="--" if name == "oracle" else "-", label=name)
    ax.set_xlabel("coverage (%)")
    ax.set_ylabel("median absolute error (deg)")
    ax.set_title("Risk–coverage: does abstaining buy accuracy?", fontsize=9)
    ax.invert_xaxis()
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", bbox_inches="tight")
    plt.close(fig)


def fig_external(out: Path, focus_json: Path, external_json: Path):
    """Internal vs external self-consistency, and detection by machine.

    The point of the left panel is the gap between the medians and the gap
    between the p90s.  A median that barely moves while the tail triples is the
    signature of a model that still works on typical images and fails outright on
    a minority — which an average would hide completely.
    """
    import json

    foc = json.loads(Path(focus_json).read_text())
    ext = json.loads(Path(external_json).read_text())
    keys = ["rot-30", "rot-15", "rot+15", "rot+30", "mirror", "gain", "scale"]
    labels = ["rot −30°", "rot −15°", "rot +15°", "rot +30°", "mirror", "gain", "crop scale"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.2),
                                 gridspec_kw={"width_ratios": [1.55, 1]})
    y = np.arange(len(keys))
    h = 0.36
    for off, src, col, name in ((+h / 2, foc, GT, "FOCUS (in-distribution)"),
                                (-h / 2, ext["overall"]["props"], PRED, "FETAL_PLANES (external)")):
        med = [src[k]["median"] for k in keys]
        p90 = [src[k]["p90"] for k in keys]
        a1.barh(y + off, p90, height=h, color=col, alpha=0.30,
                label=f"{name} — p90")
        a1.barh(y + off, med, height=h, color=col, label=f"{name} — median")
    a1.set_yticks(y, labels)
    a1.invert_yaxis()
    a1.set_xlabel("axis movement under a transformation that should not move it (deg)")
    a1.set_title("Self-consistency, no labels used", fontsize=9)
    a1.legend(fontsize=6.2, loc="lower right")

    bm = ext["by_machine"]
    machines = sorted(bm, key=lambda m: -bm[m]["n"])
    rate = [100 * bm[m]["fire_rate"] for m in machines]
    conf = [bm[m]["mean_conf"] for m in machines]
    x = np.arange(len(machines))
    a2.bar(x, rate, color="#264653", width=0.6)
    for i, (r, c, m) in enumerate(zip(rate, conf, machines)):
        a2.text(i, r + 1.5, f"{r:.0f}%\nconf {c:.2f}", ha="center", fontsize=6.5)
    a2.set_xticks(x, [f"{m}\n(n={bm[m]['n']})" for m in machines], fontsize=6.5)
    a2.set_ylim(0, 118)
    a2.set_ylabel("detector fires (%)")
    a2.set_title("Detection on unseen machines", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "external.png", bbox_inches="tight")
    plt.close(fig)


def obb_from_axes(pts: np.ndarray) -> np.ndarray:
    """Four ellipse axis endpoints -> the four corners of the oriented box."""
    c = pts[:2].mean(0)
    u = (pts[0] - pts[1]) / 2.0
    v = (pts[2] - pts[3]) / 2.0
    return np.stack([c + u + v, c + u - v, c - u - v, c - u + v])


def _poly(ax, corners, color, label=None, ls="-", lw=1.6):
    xs = list(corners[:, 0]) + [corners[0, 0]]
    ys = list(corners[:, 1]) + [corners[0, 1]]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, label=label, solid_joinstyle="round")


def fig_obb(rows, out: Path, n=4):
    """Oriented box versus the axis-aligned box the detector actually consumes.

    FOCUS ships oriented boxes; YOLOv5 takes axis-aligned ones, so stage one is
    handed the grey rectangle and stage two has to put the angle back.  The ratio
    printed on each panel is how much box area that costs.
    """
    order = np.argsort([abs(45 - (r["gt"] % 90)) for r in rows])  # most tilted first
    pick = order[:n]

    fig, axes = plt.subplots(1, n, figsize=(2.05 * n, 2.5))
    ratios = []
    for ax, i in zip(np.atleast_1d(axes), pick):
        r = rows[i]
        gt_obb = obb_from_axes(r["gt_pts"])
        pr_obb = obb_from_axes(r["pts"])
        x0, y0 = gt_obb.min(0)
        x1, y1 = gt_obb.max(0)
        aabb = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

        a_obb = np.linalg.norm(gt_obb[0] - gt_obb[1]) * np.linalg.norm(gt_obb[1] - gt_obb[2])
        a_aabb = (x1 - x0) * (y1 - y0)
        ratios.append(a_aabb / a_obb)

        ax.imshow(r["image"], cmap="gray", vmin=0, vmax=1)
        _poly(ax, aabb, "#9aa0a6", "axis-aligned (YOLO)", ls="--", lw=1.2)
        _poly(ax, gt_obb, GT, "oriented, annotated")
        _poly(ax, pr_obb, PRED, "oriented, predicted", ls="--")
        ax.scatter(r["pts"][:2, 0], r["pts"][:2, 1], s=18, c=PRED, zorder=3, lw=0)
        ax.scatter(r["pts"][2:, 0], r["pts"][2:, 1], s=16, facecolors="none",
                   edgecolors=PRED, zorder=3, lw=1.0)
        ax.set_title(f"{r['stem']} · {r['gt']:.0f}° · box area ×{ratios[-1]:.2f}",
                     fontsize=7)
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=5.6, framealpha=0.85)
    fig.suptitle("Oriented box vs the axis-aligned box the detector is given", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "oriented_boxes.png", bbox_inches="tight")
    plt.close(fig)


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, spec = _load(Path(a.ckpt))
    rows = _predict_all(model, spec, load_split(a.raw, a.split))
    fig_qualitative(rows, out)
    fig_agreement(rows, out)
    fig_risk_coverage(rows, out)
    fig_obb(rows, out)
    if Path(a.focus_json).exists() and Path(a.external_json).exists():
        fig_external(out, Path(a.focus_json), Path(a.external_json))
    print(f"wrote {len(list(out.glob('*.png')))} figures to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/FOCUS")
    p.add_argument("--split", default="test")
    p.add_argument("--ckpt", default="runs/landmarks/landmarks_best.pt")
    p.add_argument("--out", default="docs/figures")
    p.add_argument("--focus-json", dest="focus_json", default="runs/metamorphic_focus.json")
    p.add_argument("--external-json", dest="external_json", default="runs/external.json")
    main(p.parse_args())
