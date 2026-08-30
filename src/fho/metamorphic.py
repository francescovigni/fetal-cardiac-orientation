"""Metamorphic tests — validation that needs no ground truth at all.

Each test asserts a property the estimator must satisfy by construction:

* **rotation equivariance** — rotate the input by delta, the predicted axis must
  move by exactly delta;
* **mirror equivariance** — flip left-right, the axis must reflect;
* **gain and scale invariance** — brightness, contrast and zoom must not move the
  axis at all, because none of them is anatomy.

These catch coordinate-convention bugs, wrap-around bugs and augmentation leaks
that a held-out set will happily hide, and they run on unlabelled data — so they
also work as a deployment monitor on incoming scans.

    python -m fho.metamorphic --ckpt runs/landmarks/landmarks_best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import geometry as G
from .focus import load_split
from .landmarks import CropSpec, axis_angle_under, make_example


def _predictor(ckpt: Path):
    import torch
    from .landmarks import K
    from .model import LandmarkNet

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    spec = CropSpec(**state["spec"])
    model = LandmarkNet(K, state.get("width", 32))
    model.load_state_dict(state["model"])
    model.eval()

    def predict(img: np.ndarray) -> float:
        x = torch.from_numpy(np.ascontiguousarray(img))[None, None].float()
        with torch.no_grad():
            _, coords = model(x)
        return G.axis_from_landmarks(coords[0].numpy() * 4.0)["angle"]

    return predict, spec


def _as3x3(M: np.ndarray) -> np.ndarray:
    return np.vstack([M, [0.0, 0.0, 1.0]])


def expected_angle(base_angle: float, M_base: np.ndarray, M_aug: np.ndarray) -> float:
    """Where the axis must land after an augmentation.

    Derived from the actual warp matrices rather than from a hardcoded sign
    convention.  The first version of this test asserted ``base + delta`` and
    reported errors of exactly twice delta on every image — the unmistakable
    signature of a flipped sign in the *test*, not in the model.  Computing the
    expectation from the transform removes the convention from the test entirely.
    """
    T = _as3x3(M_aug) @ np.linalg.inv(_as3x3(M_base))
    return axis_angle_under(T[:2], base_angle)


def run(ckpt: Path, raw: Path, split: str = "test",
        deltas=(-30, -15, 15, 30), n: int | None = None, as_dict: bool = False):
    predict, spec = _predictor(ckpt)
    samples = load_split(raw, split)[:n]

    rot_res = {d: [] for d in deltas}
    flip_res, gain_res, scale_res = [], [], []

    for s in samples:
        base_ex = make_example(s, spec)
        base = predict(base_ex["image"])
        M0 = base_ex["matrix"]

        for d in deltas:
            ex = make_example(s, spec, rot_deg=d)
            got = predict(ex["image"])
            rot_res[d].append(G.signed_angdiff_axial(
                got, expected_angle(base, M0, ex["matrix"])))

        ex = make_example(s, spec, flip=True)
        got = predict(ex["image"])
        flip_res.append(G.signed_angdiff_axial(
            got, expected_angle(base, M0, ex["matrix"])))

        ex = make_example(s, spec, gain=(1.4, -0.05))
        gain_res.append(G.signed_angdiff_axial(predict(ex["image"]), base))

        big = CropSpec(size=spec.size, margin=spec.margin + 0.25)
        ex = make_example(s, big)
        scale_res.append(G.signed_angdiff_axial(predict(ex["image"]), base))

    measured = {f"rot{d:+d}": rot_res[d] for d in deltas}
    measured.update(mirror=flip_res, gain=gain_res, scale=scale_res)
    if as_dict:
        return {k: dict(median=float(np.median(np.abs(v))),
                        p90=float(np.percentile(np.abs(v), 90)),
                        max=float(np.max(np.abs(v)))) for k, v in measured.items()}

    def line(name, v, tol):
        v = np.abs(np.asarray(v, float))
        ok = "PASS" if np.percentile(v, 90) <= tol else "FAIL"
        return (f"  {ok}  {name:28s} median {np.median(v):6.2f}\u00b0  "
                f"p90 {np.percentile(v, 90):6.2f}\u00b0  max {v.max():6.2f}\u00b0  "
                f"(tol p90 <= {tol}\u00b0)")

    out = [f"metamorphic tests \u2014 {len(samples)} images from '{split}', no labels used", ""]
    for d in deltas:
        out.append(line(f"rotation equivariance {d:+d}\u00b0", rot_res[d], 5.0))
    out.append(line("mirror equivariance", flip_res, 5.0))
    out.append(line("gain invariance", gain_res, 2.0))
    out.append(line("crop-scale invariance", scale_res, 5.0))
    return "\n".join(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/landmarks/landmarks_best.pt")
    p.add_argument("--raw", default="data/raw/FOCUS")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    print(run(Path(a.ckpt), Path(a.raw), a.split, n=a.n))
    if a.json:
        import json
        Path(a.json).write_text(json.dumps(
            run(Path(a.ckpt), Path(a.raw), a.split, n=a.n, as_dict=True), indent=1))
        print(f"wrote {a.json}")
