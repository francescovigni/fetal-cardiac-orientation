"""Validation harness — the part that answers "how do you know it works?".

Five things, none of which is a single accuracy number:

1. **Agreement, not accuracy.**  Bland-Altman bias and 95 % limits of agreement
   against the reference, plus ICC(2,1).  Mean error hides systematic bias, which
   is the failure that matters clinically.
2. **Circular statistics.**  Angles do not average linearly and the heart axis is
   axial, so everything goes through the doubled-angle representation.
3. **Stratification.**  Error by shape roundness, by heart size, by image source.
   A global number hides where it fails, and roundness is the covariate that
   provably breaks orientation estimators.
4. **Risk-coverage.**  Error as a function of how much you abstain, using a
   confidence score.  If error does not fall as coverage drops, the confidence
   estimate is decorative.
5. **Stage attribution.**  The same model evaluated on ground-truth crops and on
   YOLO crops.  The gap is what stage 1 costs the final angle.

    python -m fho.evaluate --split test --ckpt runs/landmarks/landmarks_best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import geometry as G

# --------------------------------------------------------------------------- #
# agreement statistics                                                         #
# --------------------------------------------------------------------------- #


def bland_altman(pred, ref) -> dict:
    """Bias and 95 % limits of agreement on the signed axial difference."""
    d = G.signed_angdiff_axial(pred, ref)
    bias, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    return dict(
        bias_deg=bias,
        sd_deg=sd,
        loa_lower=bias - 1.96 * sd,
        loa_upper=bias + 1.96 * sd,
        n=int(len(d)),
    )


def icc21(a, b) -> float:
    """ICC(2,1), two-way random effects, absolute agreement, single measure.

    Computed on the *unwrapped* angles, which is valid here because the axial
    differences are small; if a dataset ever spans the wrap, use the circular
    correlation instead and say so.
    """
    x = np.stack([np.asarray(a, float), np.asarray(b, float)], 1)
    n, k = x.shape
    gm = x.mean()
    ms_r = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    ms_e = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum() / (
        (n - 1) * (k - 1)
    )
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if denom else float("nan")


def bootstrap_ci(values, stat=np.median, n_boot=2000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    boot = [stat(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# --------------------------------------------------------------------------- #
# stratification and risk-coverage                                             #
# --------------------------------------------------------------------------- #


def stratify(errors, covariate, edges, name="covariate") -> str:
    e, c = np.asarray(errors, float), np.asarray(covariate, float)
    lines = [f"  error by {name}:"]
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        k = (c >= lo) & (c < hi)
        if k.sum() == 0:
            continue
        lines.append(
            f"    {lo:6.2f}–{hi:6.2f}  n={k.sum():3d}  "
            f"median {np.median(e[k]):6.2f}°  p90 {np.percentile(e[k], 90):6.2f}°"
        )
    return "\n".join(lines)


def risk_coverage(errors, confidence, steps=10) -> str:
    """Error against retained fraction, sorted by descending confidence."""
    e, c = np.asarray(errors, float), np.asarray(confidence, float)
    order = np.argsort(-c)
    e = e[order]
    lines = ["  risk-coverage (abstain on the least confident):", "    coverage  median   p90"]
    for f in np.linspace(1.0, 0.3, steps):
        m = max(int(round(f * len(e))), 1)
        lines.append(f"    {f:7.0%}  {np.median(e[:m]):6.2f}° {np.percentile(e[:m], 90):6.2f}°")
    return "\n".join(lines)


def report(pred, ref, covariates: dict | None = None, confidence=None, title="") -> str:
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    err = G.angdiff_axial(pred, ref)
    ba = bland_altman(pred, ref)
    lo, hi = bootstrap_ci(err)
    out = [
        f"== {title} ==  n={len(err)}",
        "",
        f"  median |error|   {np.median(err):6.2f}°   95% CI [{lo:.2f}, {hi:.2f}]",
        f"  mean   |error|   {err.mean():6.2f}°",
        f"  p90    |error|   {np.percentile(err, 90):6.2f}°",
        f"  max    |error|   {err.max():6.2f}°",
        f"  fraction >10°    {100 * np.mean(err > 10):5.1f}%",
        "",
        f"  Bland-Altman     bias {ba['bias_deg']:+.2f}°   "
        f"LoA [{ba['loa_lower']:+.2f}, {ba['loa_upper']:+.2f}]",
        f"  ICC(2,1)         {icc21(pred, ref):.4f}",
        f"  circular SD of the signed difference  "
        f"{G.circ_sd_axial(G.signed_angdiff_axial(pred, ref)):.2f}°",
        "",
    ]
    if covariates:
        for name, (vals, edges) in covariates.items():
            out.append(stratify(err, vals, edges, name))
            out.append("")
    if confidence is not None:
        out.append(risk_coverage(err, confidence))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# entry point                                                                  #
# --------------------------------------------------------------------------- #


def _predict(ckpt: Path, raw: Path, split: str, source: str) -> dict:
    import torch

    from .focus import load_split
    from .landmarks import CropSpec, K, make_example
    from .model import LandmarkNet

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    spec = CropSpec(**state["spec"])
    model = LandmarkNet(K, state.get("width", 32))
    model.load_state_dict(state["model"])
    model.eval()

    preds, refs, aniso, size, conf, disagree = [], [], [], [], [], []
    for s in load_split(raw, split):
        ex = make_example(s, spec)
        x = torch.from_numpy(ex["image"])[None, None]
        with torch.no_grad():
            axis, coords = model(x)
        pts = coords[0].numpy() * 4.0
        d = G.axis_from_landmarks(pts)
        direct = G.decode_axial(axis[0].numpy())
        preds.append(d["angle"])
        refs.append(ex["gt_angle"])
        aniso.append(ex["anisotropy"])
        size.append(s.cardiac.a)
        disagree.append(d["disagreement"])
        # confidence = the two heads agreeing.  Negated so that larger is more
        # confident, which is what the risk-coverage sort expects.
        conf.append(-float(G.angdiff_axial(d["angle"], direct)))
    return dict(pred=preds, ref=refs, aniso=aniso, size=size, conf=conf, disagree=disagree)


def main(a):
    raw = Path(a.raw)
    if a.ckpt and Path(a.ckpt).exists():
        r = _predict(Path(a.ckpt), raw, a.split, a.source)
        print(
            report(
                r["pred"],
                r["ref"],
                covariates={
                    "roundness b/a": (r["aniso"], [0.0, 0.6, 0.75, 0.9, 1.01]),
                    "heart semi-major (px)": (r["size"], [0, 60, 90, 120, 400]),
                },
                confidence=r["conf"],
                title=f"landmark model / {a.split} / crops from {a.source}",
            )
        )
        print()
        print(
            report(
                r["pred"],
                r["ref"],
                confidence=-np.array(r["disagree"]),
                title="same predictions, confidence = major/minor axis agreement",
            )
        )
        Path(a.out).write_text(json.dumps(r, indent=1))
    else:
        from .baselines import run, summarise

        print("no checkpoint given — classical baselines only\n")
        print(summarise(run(raw, a.split)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/FOCUS")
    p.add_argument("--split", default="test")
    p.add_argument("--ckpt", default="runs/landmarks/landmarks_best.pt")
    p.add_argument("--source", default="gt", choices=["gt", "yolo"])
    p.add_argument("--out", default="runs/landmarks/eval.json")
    main(p.parse_args())
