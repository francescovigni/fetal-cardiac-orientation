"""Can you get the orientation from an existing pipeline, without training for it?

If a segmentation of the heart already exists — from a detector, a segmenter, or a
clinician's outline — the orientation is available in closed form from the
second-order central moments of that mask.  No training, no labels, microseconds.

The obvious objection is that a real pipeline's mask is not the annotation.  So
this module degrades the ground-truth mask in the ways a segmenter actually fails
and measures how the angle error grows as mask quality drops:

* **erode / dilate** — systematic under- and over-segmentation;
* **boundary noise** — a ragged contour at roughly the right place;
* **bite** — a chunk missing, as when a chamber is lost to shadowing;
* **blob** — a spurious extra component, as when neighbouring tissue is included.

Each corruption is scored by Dice against the true mask, so the output is a
transfer curve: *given a segmenter of quality D, expect an angle error of E*.
That is the number to quote when someone asks whether their existing pipeline is
good enough, and it is measurable without training anything.

    python -m fho.no_training --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import geometry as G
from .focus import load_split

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# corruptions                                                                  #
# --------------------------------------------------------------------------- #

def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def erode(m, k):
    return cv2.erode(m, _disk(int(k)))


def dilate(m, k):
    return cv2.dilate(m, _disk(int(k)))


def boundary_noise(m, k, rng=RNG):
    """Perturb the contour by a smooth random field — a ragged but centred mask."""
    h, w = m.shape
    f = rng.normal(0, 1, (h // 8 + 1, w // 8 + 1)).astype(np.float32)
    f = cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)
    f = cv2.GaussianBlur(f, (0, 0), 6)
    f = f / (np.abs(f).max() + 1e-9)
    return ((m.astype(np.float32) + k * f) > 0.5).astype(np.uint8)


def bite(m, frac, rng=RNG):
    """Remove a wedge of the mask, as when part of the heart is shadowed out."""
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return m
    c = np.array([xs.mean(), ys.mean()])
    r = float(np.sqrt(((xs - c[0]) ** 2 + (ys - c[1]) ** 2).max()))
    ang = rng.uniform(0, 2 * np.pi)
    d = np.array([np.cos(ang), np.sin(ang)])
    p = c + d * r * (1.0 - frac)
    out = m.copy()
    cv2.circle(out, (int(p[0]), int(p[1])), int(r * frac), 0, -1)
    return out


def blob(m, frac, rng=RNG):
    """Add a spurious component near the mask, as when neighbouring tissue leaks in."""
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return m
    c = np.array([xs.mean(), ys.mean()])
    r = float(np.sqrt(((xs - c[0]) ** 2 + (ys - c[1]) ** 2).max()))
    ang = rng.uniform(0, 2 * np.pi)
    p = c + np.array([np.cos(ang), np.sin(ang)]) * r * 1.15
    out = m.copy()
    cv2.circle(out, (int(p[0]), int(p[1])), max(int(r * frac), 1), 1, -1)
    return out


CORRUPTIONS = {
    "erode": (erode, [0, 2, 5, 9, 14, 20]),
    "dilate": (dilate, [0, 2, 5, 9, 14, 20]),
    "boundary noise": (boundary_noise, [0.0, 0.2, 0.4, 0.7, 1.0, 1.4]),
    "bite": (bite, [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]),
    "blob": (blob, [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]),
}


def dice(a, b) -> float:
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


# --------------------------------------------------------------------------- #
# estimators that require no training                                          #
# --------------------------------------------------------------------------- #

def cleanup(mask: np.ndarray, open_radius: int = 3) -> np.ndarray:
    """Two lines that decide whether this approach works at all.

    Keep only the largest connected component, then morphologically open.  Second
    moments are computed over *everything* that is set, so one spurious blob of
    neighbouring tissue drags the principal axis toward it — and, worse, widens
    the eigengap, so the estimate becomes more confident as it becomes wrong.
    Dropping the smaller components removes that failure entirely; the opening
    then takes the ragged edge off a noisy contour.
    """
    m = (mask > 0).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 2:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lbl == k).astype(np.uint8)
    if open_radius > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _disk(open_radius))
    return m


def estimate(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return dict(moments=np.nan, minarea=np.nan, cleaned=np.nan,
                    se=np.inf, se_cleaned=np.inf, aniso=1.0)
    pts = np.stack([xs, ys], 1).astype(float)
    mom = G.pca_axis(pts)
    mar = G.minarea_axis(pts)

    mc = cleanup(mask)
    ysc, xsc = np.nonzero(mc)
    if len(xsc) >= 10:
        momc = G.pca_axis(np.stack([xsc, ysc], 1).astype(float))
    else:
        momc = dict(angle=mom["angle"], angle_se_deg=mom["angle_se_deg"])

    return dict(moments=mom["angle"], minarea=mar["angle"],
                cleaned=momc["angle"], se=mom["angle_se_deg"],
                se_cleaned=momc["angle_se_deg"], aniso=mom["anisotropy"])


# --------------------------------------------------------------------------- #
# experiment                                                                   #
# --------------------------------------------------------------------------- #

def run(raw: Path, split: str) -> list[dict]:
    rows = []
    for s in load_split(raw, split):
        if "cardiac" not in s.mask_paths:
            continue
        m0 = (np.array(Image.open(s.mask_paths["cardiac"]).convert("L")) > 127).astype(np.uint8)
        gt = s.cardiac.theta
        for name, (fn, levels) in CORRUPTIONS.items():
            for lv in levels:
                m = m0 if lv == 0 else fn(m0, lv)
                if m.sum() < 50:
                    continue
                e = estimate(m)
                rows.append(dict(
                    stem=s.stem, corruption=name, level=float(lv),
                    dice=dice(m, m0),
                    err_moments=float(G.angdiff_axial(e["moments"], gt)),
                    err_minarea=float(G.angdiff_axial(e["minarea"], gt)),
                    err_cleaned=float(G.angdiff_axial(e["cleaned"], gt)),
                    se=float(e["se"]), se_cleaned=float(e["se_cleaned"]),
                    aniso=float(e["aniso"]),
                ))
    return rows


def report(rows) -> str:
    import collections

    out = ["Orientation from an existing segmentation — no training, no labels", ""]
    out.append("  Angle error against mask quality (Dice vs the true mask):")
    out.append(f"    {'Dice':>10s} {'n':>5s} {'moments (raw)':>18s} "
               f"{'moments (cleaned)':>18s} {'min-area rect':>18s}")
    bins = [(0.98, 1.01), (0.95, 0.98), (0.90, 0.95), (0.80, 0.90),
            (0.70, 0.80), (0.0, 0.70)]
    d = np.array([r["dice"] for r in rows])
    for lo, hi in bins:
        k = (d >= lo) & (d < hi)
        if k.sum() == 0:
            continue
        em = np.array([r["err_moments"] for r in rows])[k]
        ec = np.array([r["err_cleaned"] for r in rows])[k]
        ea = np.array([r["err_minarea"] for r in rows])[k]
        out.append(f"    {lo:.2f}–{hi:.2f} {k.sum():5d} "
                   f"{np.median(em):7.2f}° p90 {np.percentile(em, 90):6.2f}° "
                   f"{np.median(ec):7.2f}° p90 {np.percentile(ec, 90):6.2f}° "
                   f"{np.median(ea):7.2f}° p90 {np.percentile(ea, 90):6.2f}°")

    out += ["", "  By failure mode, at the harshest level applied:"]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["corruption"]].append(r)
    for name, rs in by.items():
        worst = max(x["level"] for x in rs)
        sel = [x for x in rs if x["level"] == worst]
        em = np.array([x["err_moments"] for x in sel])
        ec = np.array([x["err_cleaned"] for x in sel])
        out.append(f"    {name:16s} level {worst:>5g}  "
                   f"Dice {np.median([x['dice'] for x in sel]):.2f}  "
                   f"raw {np.median(em):6.2f}° (p90 {np.percentile(em, 90):5.1f}°)  "
                   f"cleaned {np.median(ec):6.2f}° (p90 {np.percentile(ec, 90):5.1f}°)")

    clean = [r for r in rows if r["level"] == 0 and r["corruption"] == "erode"]
    if clean:
        em = np.array([r["err_moments"] for r in clean])
        ea = np.array([r["err_minarea"] for r in clean])
        out += ["", f"  On the undamaged mask (n={len(clean)}): "
                    f"moments median {np.median(em):.2f}°, "
                    f"min-area median {np.median(ea):.2f}° "
                    f"({100*np.mean(ea > 10):.0f}% beyond 10°)"]

    for label, sek, ek in (("raw", "se", "err_moments"),
                           ("cleaned", "se_cleaned", "err_cleaned")):
        se = np.array([r[sek] for r in rows])
        err = np.array([r[ek] for r in rows])
        ok = np.isfinite(se) & np.isfinite(err)
        if ok.sum() > 10:
            c = np.corrcoef(np.log10(se[ok] + 1e-9), err[ok])[0, 1]
            out += [f"  eigengap standard error vs actual error ({label}): r = {c:+.2f}"]
    return "\n".join(out)


def main(a):
    rows = run(Path(a.raw), a.split)
    print(report(rows))
    if a.json:
        import json
        Path(a.json).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/FOCUS")
    p.add_argument("--split", default="test")
    p.add_argument("--json", default="runs/no_training.json")
    main(p.parse_args())
