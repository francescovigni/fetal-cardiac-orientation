"""Closed-loop consistency: detect, de-rotate by the estimate, detect again.

The metamorphic suite in :mod:`fho.metamorphic` tests each stage on ground-truth
crops.  This composes the **deployed** path instead:

    detect  ->  estimate theta  ->  warp the image so the heart is axis-aligned
            ->  detect again    ->  estimate theta again

If the first estimate were exact, the second must read zero.  The residual is
therefore a second-order, label-free measure of the whole pipeline, and it
exercises three things the unit-level tests do not: that the detector still fires
on a **resampled** image (a double warp blurs speckle), that it re-localises the
same structure, and that the orientation estimator is **idempotent** — that its
own output is a fixed point of the transformation it defines.

**What this does not prove.**  A degenerate estimator that always returns zero
passes this test trivially: de-rotating by zero is the identity.  The round trip
is therefore a *necessary* condition, meaningful only alongside the rotation
equivariance test, which a constant predictor fails by construction.  Both are
reported together for that reason.

A control run de-rotates by a **random** angle instead of the estimate.  The
residual there must come back as that angle, which verifies the warp algebra
rather than the model — if the control fails, the measurement is broken, not the
network.

    python -m fho.roundtrip --n 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from . import geometry as G
from .focus import load_split
from .landmarks import CropSpec, affine_crop_matrix, axis_angle_under


def derotate(
    image: np.ndarray, cx: float, cy: float, angle_deg: float, half_size: float, out: int
) -> tuple[np.ndarray, np.ndarray]:
    """Warp so that ``angle_deg`` about (cx, cy) becomes horizontal in the output."""
    M = affine_crop_matrix(cx, cy, half_size, out, rot_deg=angle_deg)
    warped = cv2.warpAffine(image, M, (out, out), flags=cv2.INTER_LINEAR, borderValue=0.0)
    return warped, M


def derotate_full(
    image: np.ndarray, cx: float, cy: float, angle_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the whole image about (cx, cy), preserving its dimensions.

    The detector must see the same framing statistics on both passes.  An earlier
    version of this module re-detected on the tight 192 px crop, where the heart
    fills the frame; the re-detection rate collapsed from 100 % to 10 %, which
    measured the scale and context shift rather than anything about rotation.
    Rotating the full image keeps rotation the only variable.
    """
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle_deg), 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0), M


def residual_after_derotation(image, cx, cy, angle_deg, half_size, out, measure) -> dict:
    """De-rotate by ``angle_deg``, measure again, and report the residual.

    ``measure`` maps a crop to an axial angle in crop coordinates.  The expected
    reading after de-rotating by the true angle is the image of that angle under
    the warp, which is computed from the matrix rather than assumed to be zero.
    """
    warped, M = derotate(image, cx, cy, angle_deg, half_size, out)
    got = measure(warped)
    expected = axis_angle_under(M, angle_deg)
    return dict(
        measured=got,
        expected=expected,
        residual=float(G.angdiff_axial(got, expected)),
        warped=warped,
    )


# --------------------------------------------------------------------------- #
# the real pipeline: YOLO -> landmarks -> de-rotate -> YOLO -> landmarks        #
# --------------------------------------------------------------------------- #


def _load(yolo_weights: Path, repo: Path, ckpt: Path):
    import torch

    from .landmarks import K
    from .model import LandmarkNet

    det = torch.hub.load(str(repo), "custom", path=str(yolo_weights), source="local", verbose=False)
    det.conf = 0.25
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = LandmarkNet(K, state.get("width", 32))
    model.load_state_dict(state["model"])
    model.eval()
    return det, model, CropSpec(**state["spec"])


def _detect(det, image_u8: np.ndarray):
    r = det(image_u8).xyxy[0].cpu().numpy()
    c = r[r[:, 5] == 0]
    if len(c) == 0:
        return None, 0.0
    b = c[c[:, 4].argmax()]
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(b[4])


def _measure(model, spec):
    import torch

    def measure(crop: np.ndarray) -> float:
        with torch.no_grad():
            _, coords = model(torch.from_numpy(crop)[None, None].float())
        return G.axis_from_landmarks(coords[0].numpy() * 4.0)["angle"]

    return measure


def run(
    raw: Path, split: str, yolo: Path, repo: Path, ckpt: Path, n: int | None = None, seed: int = 0
) -> list[dict]:
    det, model, spec = _load(yolo, repo, ckpt)
    measure = _measure(model, spec)
    rng = np.random.default_rng(seed)
    rows = []

    for s in load_split(raw, split)[:n]:
        img_u8 = cv2.imread(str(s.image_path), cv2.IMREAD_GRAYSCALE)
        img = img_u8.astype(np.float32) / 255.0

        box, conf = _detect(det, img_u8)
        if box is None:
            rows.append(dict(stem=s.stem, fired=False))
            continue
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        r = max(x1 - x0, y1 - y0) / 2 * (1.0 + spec.margin)

        crop, M0 = derotate(img, cx, cy, 0.0, r, spec.size)
        theta1 = measure(crop)
        theta1_img = axis_angle_under(np.linalg.inv(np.vstack([M0, [0, 0, 1]]))[:2], theta1)

        # --- the round trip: rotate the FULL image by the estimate, redo both stages ---
        full, R1 = derotate_full(img, cx, cy, theta1_img)
        full_u8 = (np.clip(full, 0, 1) * 255).astype(np.uint8)
        box2, conf2 = _detect(det, cv2.cvtColor(full_u8, cv2.COLOR_GRAY2BGR))

        target = axis_angle_under(R1, theta1_img)  # what a perfect estimate reads back
        if box2 is not None:
            bx, by = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
            br = max(box2[2] - box2[0], box2[3] - box2[1]) / 2 * (1.0 + spec.margin)
            crop2, M2 = derotate(full, bx, by, 0.0, br, spec.size)
            theta2 = axis_angle_under(np.linalg.inv(np.vstack([M2, [0, 0, 1]]))[:2], measure(crop2))
            expect_c = R1[:, :2] @ np.array([cx, cy]) + R1[:, 2]
            centre_shift = float(np.hypot(bx - expect_c[0], by - expect_c[1]))
        else:
            theta2, centre_shift = float("nan"), float("nan")

        # --- control: rotate by a random angle instead of the estimate ---
        ctrl_angle = float(rng.uniform(-60, 60))
        ctrl, Rc = derotate_full(img, cx, cy, ctrl_angle)
        cc, Mc = derotate(ctrl, *(Rc[:, :2] @ np.array([cx, cy]) + Rc[:, 2]), 0.0, r, spec.size)
        ctrl_measured = axis_angle_under(np.linalg.inv(np.vstack([Mc, [0, 0, 1]]))[:2], measure(cc))
        ctrl_residual = float(G.angdiff_axial(ctrl_measured, axis_angle_under(Rc, theta1_img)))

        rows.append(
            dict(
                stem=s.stem,
                fired=True,
                conf=conf,
                theta1=theta1_img,
                theta2=theta2,
                residual=float(G.angdiff_axial(theta2, target)),
                refired=box2 is not None,
                conf2=conf2,
                centre_shift=centre_shift,
                control_residual=ctrl_residual,
                gt_error=float(G.angdiff_axial(theta1_img, s.cardiac.theta)),
            )
        )
    return rows


def report(rows) -> str:
    fired = [r for r in rows if r["fired"]]
    out = [f"Round trip: detect -> orient -> de-rotate -> detect -> orient   n={len(rows)}", ""]
    if not fired:
        return "\n".join(out + ["  detector never fired"])

    refired = [r for r in fired if r["refired"]]
    out.append(
        f"  first detection      {len(fired)}/{len(rows)} "
        f"({100 * len(fired) / len(rows):.0f}%)  mean conf "
        f"{np.mean([r['conf'] for r in fired]):.3f}"
    )
    out.append(
        f"  re-detection after warp  {len(refired)}/{len(fired)} "
        f"({100 * len(refired) / len(fired):.0f}%)  mean conf "
        f"{np.mean([r['conf2'] for r in refired]):.3f}"
    )
    if refired:
        cs = np.array([r["centre_shift"] for r in refired])
        out.append(
            f"  re-detected centre displacement  median {np.median(cs):.1f} px "
            f"from where the warp puts it"
        )

    # Re-detection against the rotation actually applied.  This is the stratification
    # that turns "52 % re-detected" into a diagnosis: the detector holds until the
    # rotation leaves the range it was augmented over, then falls off a cliff.
    def applied(theta):
        theta %= 180.0
        return min(theta, 180.0 - theta)

    ang = np.array([applied(r["theta1"]) for r in fired])
    ok = np.array([r["refired"] for r in fired])
    out += ["", "  re-detection vs the rotation the de-rotation applies:"]
    for lo, hi in ((0, 30), (30, 45), (45, 70), (70, 91)):
        k = (ang >= lo) & (ang < hi)
        if k.sum():
            out.append(
                f"    |rot| {lo:2d}-{hi:2d}°   n={k.sum():3d}   re-detected {100 * ok[k].mean():5.1f}%"
            )
    if len(set(ok)) > 1:
        out.append(
            f"    point-biserial correlation r = {np.corrcoef(ang, ok.astype(float))[0, 1]:+.2f}"
        )

    res = np.array([r["residual"] for r in fired if np.isfinite(r["residual"])])
    ctl = np.array([r["control_residual"] for r in fired])
    gte = np.array([r["gt_error"] for r in fired])
    out += [
        "",
        f"  {'':22s} {'median':>8s} {'p90':>8s} {'max':>8s}",
        f"  {'round-trip residual':22s} {np.median(res):7.2f}° "
        f"{np.percentile(res, 90):7.2f}° {res.max():7.2f}°",
        f"  {'control (random warp)':22s} {np.median(ctl):7.2f}° "
        f"{np.percentile(ctl, 90):7.2f}° {ctl.max():7.2f}°",
        f"  {'error vs annotation':22s} {np.median(gte):7.2f}° "
        f"{np.percentile(gte, 90):7.2f}° {gte.max():7.2f}°",
        "",
        "  A constant-zero estimator passes the round trip trivially, so this is a",
        "  necessary condition only — read it with the rotation equivariance test.",
    ]
    return "\n".join(out)


def main(a):
    rows = run(Path(a.raw), a.split, Path(a.yolo), Path(a.repo), Path(a.ckpt), a.n)
    print(report(rows))
    if a.json:
        import json

        Path(a.json).write_text(
            json.dumps([{k: v for k, v in r.items() if k != "warped"} for r in rows], indent=1)
        )
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/FOCUS")
    p.add_argument("--split", default="test")
    p.add_argument("--yolo", default="runs/yolo/focus/weights/best.pt")
    p.add_argument("--repo", default="yolov5")
    p.add_argument("--ckpt", default="runs/landmarks/landmarks_best.pt")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--json", default="runs/roundtrip.json")
    main(p.parse_args())
