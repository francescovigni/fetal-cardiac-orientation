"""Classical orientation baselines, run on the ground-truth cardiac masks.

These are what the learned landmark model has to beat.  They need no training and
no GPU, so they also serve as the first end-to-end sanity check of the geometry
code:  mask -> angle -> comparison against the annotated ellipse angle.

    python -m fho.baselines --raw data/raw/FOCUS --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from . import geometry as G
from .focus import load_split


def run(raw: Path, split: str, label: str = "cardiac") -> dict:
    rows = []
    for s in load_split(raw, split):
        if label not in s.mask_paths:
            continue
        m = np.array(Image.open(s.mask_paths[label]).convert("L"), float) / 255.0
        gt = s.ellipses[label].theta

        moments = G.mask_axis(m)
        ys, xs = np.nonzero(m > 0.5)
        pts = np.stack([xs, ys], 1).astype(float)
        mar = G.minarea_axis(pts) if len(pts) >= 3 else dict(angle=np.nan)

        rows.append(dict(
            stem=s.stem, gt=gt,
            moments=moments["angle"], minarea=mar["angle"],
            err_moments=float(G.angdiff_axial(moments["angle"], gt)),
            err_minarea=float(G.angdiff_axial(mar["angle"], gt)),
            anisotropy=moments["anisotropy"],
            angle_se=moments["angle_se_deg"],
            ellipse_aniso=s.ellipses[label].anisotropy,
        ))
    return {"split": split, "label": label, "rows": rows}


def summarise(res: dict) -> str:
    rows = res["rows"]
    out = [f"{res['label']} / {res['split']}  n={len(rows)}", ""]
    for name in ("moments", "minarea"):
        e = np.array([r[f"err_{name}"] for r in rows])
        out.append(f"  {name:9s} median {np.median(e):6.2f}°   mean {e.mean():6.2f}°   "
                   f"p90 {np.percentile(e, 90):6.2f}°   max {e.max():6.2f}°   "
                   f"|err|>10° {100*np.mean(e > 10):4.1f}%")
    a = np.array([r["ellipse_aniso"] for r in rows])
    e = np.array([r["err_moments"] for r in rows])
    out += ["", "  moments error vs shape roundness (b/a):"]
    for lo, hi in ((0.0, 0.6), (0.6, 0.75), (0.75, 0.9), (0.9, 1.01)):
        k = (a >= lo) & (a < hi)
        if k.sum():
            out.append(f"    b/a {lo:.2f}–{hi:.2f}  n={k.sum():3d}  "
                       f"median {np.median(e[k]):6.2f}°   p90 {np.percentile(e[k], 90):6.2f}°")
    return "\n".join(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/FOCUS")
    p.add_argument("--split", default="test")
    p.add_argument("--label", default="cardiac")
    a = p.parse_args()
    print(summarise(run(Path(a.raw), a.split, a.label)))
