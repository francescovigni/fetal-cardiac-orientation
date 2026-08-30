"""External evaluation on FETAL_PLANES_DB, with no labels of any kind.

FOCUS has orientation ground truth; almost no other public fetal dataset does.
That does not prevent an external generalisation test, because the properties the
estimator must satisfy do not need labels:

* rotate the input by delta -> the predicted axis must move by exactly delta;
* mirror the input          -> the axis must reflect;
* change gain or contrast   -> the axis must not move, none of it is anatomy;
* widen the crop            -> the axis must not move.

Running the same suite on FOCUS and on a different hospital's images turns
"we did not evaluate generalisation" into a measured number.  FETAL_PLANES_DB
also records the **ultrasound machine** per image, so the result can be split by
manufacturer, which is the axis that matters for a machine-agnostic claim.

    python -m fho.external --n 600 --by-machine

Data: FETAL_PLANES_DB, Zenodo 3904280, CC-BY-4.0, 12,400 images, of which 1,718
are labelled 'Fetal thorax' (the four-chamber plane).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import geometry as G
from .landmarks import CropSpec, affine_crop_matrix, axis_angle_under

THORAX = "Fetal thorax"


# --------------------------------------------------------------------------- #
# data                                                                         #
# --------------------------------------------------------------------------- #

def load_index(root: Path, plane: str = THORAX) -> list[dict]:
    with open(root / "FETAL_PLANES_DB_data.csv") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    out = []
    for r in rows:
        if r["Plane"].strip() != plane:
            continue
        name = r["Image_name"].strip()
        p = root / "Images" / f"{name}.png"
        if p.exists():
            out.append(dict(path=p, machine=r["US_Machine"].strip(),
                            operator=r["Operator"].strip(),
                            patient=r["Patient_num"].strip()))
    return out


# --------------------------------------------------------------------------- #
# stage 1: detection, which also gives the firing rate for free                #
# --------------------------------------------------------------------------- #

def load_detector(weights: Path, repo: Path):
    import torch
    model = torch.hub.load(str(repo), "custom", path=str(weights),
                           source="local", verbose=False)
    model.conf = 0.25
    return model


def detect_cardiac(det, path: Path):
    """Highest-confidence cardiac box, or None if the detector does not fire."""
    r = det(str(path)).xyxy[0].cpu().numpy()
    c = r[r[:, 5] == 0]
    if len(c) == 0:
        return None, 0.0
    b = c[c[:, 4].argmax()]
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(b[4])


# --------------------------------------------------------------------------- #
# stage 2 + the label-free properties                                          #
# --------------------------------------------------------------------------- #

def load_landmark_model(ckpt: Path):
    import torch
    from .landmarks import K
    from .model import LandmarkNet

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = LandmarkNet(K, state.get("width", 32))
    m.load_state_dict(state["model"])
    m.eval()
    return m, CropSpec(**state["spec"])


def _warp(img, cx, cy, r, size, rot_deg=0.0, flip=False, gain=(1.0, 0.0)):
    M = affine_crop_matrix(cx, cy, r, size, rot_deg)
    a = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=0.0)
    if flip:
        a = np.ascontiguousarray(a[:, ::-1])
        M = np.array([[-1.0, 0.0, size - 1.0], [0.0, 1.0, 0.0]]) @ np.vstack([M, [0, 0, 1]])
    if gain != (1.0, 0.0):
        a = np.clip(a * gain[0] + gain[1], 0.0, 1.0)
    return a.astype(np.float32), M


def predict_angle(model, crop):
    import torch
    with torch.no_grad():
        _, coords = model(torch.from_numpy(crop)[None, None])
    return G.axis_from_landmarks(coords[0].numpy() * 4.0)["angle"]


def expected(base_angle, M_base, M_aug):
    T = np.vstack([M_aug, [0, 0, 1]]) @ np.linalg.inv(np.vstack([M_base, [0, 0, 1]]))
    return axis_angle_under(T[:2], base_angle)


def run(items, det, model, spec, deltas=(-30, -15, 15, 30)) -> list[dict]:
    out = []
    for it in items:
        box, conf = detect_cardiac(det, it["path"])
        if box is None:
            out.append(dict(**it, fired=False))
            continue
        img = cv2.imread(str(it["path"]), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        r = max(x1 - x0, y1 - y0) / 2 * (1.0 + spec.margin)

        c0, M0 = _warp(img, cx, cy, r, spec.size)
        base = predict_angle(model, c0)

        rec = dict(**it, fired=True, conf=conf, angle=base)
        for d in deltas:
            c, M = _warp(img, cx, cy, r, spec.size, rot_deg=d)
            rec[f"rot{d:+d}"] = abs(G.signed_angdiff_axial(
                predict_angle(model, c), expected(base, M0, M)))
        c, M = _warp(img, cx, cy, r, spec.size, flip=True)
        rec["mirror"] = abs(G.signed_angdiff_axial(
            predict_angle(model, c), expected(base, M0, M)))
        c, _ = _warp(img, cx, cy, r, spec.size, gain=(1.4, -0.05))
        rec["gain"] = abs(G.signed_angdiff_axial(predict_angle(model, c), base))
        c, _ = _warp(img, cx, cy, r * 1.25, spec.size)
        rec["scale"] = abs(G.signed_angdiff_axial(predict_angle(model, c), base))
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# reporting                                                                    #
# --------------------------------------------------------------------------- #

PROPS = [("rot-30", "rot-30"), ("rot-15", "rot-15"), ("rot+15", "rot+15"),
         ("rot+30", "rot+30"), ("mirror", "mirror"), ("gain", "gain"),
         ("scale", "scale")]


def _table(rows, title) -> str:
    fired = [r for r in rows if r["fired"]]
    lines = [f"{title}   n={len(rows)}   detector fired on {len(fired)} "
             f"({100*len(fired)/max(len(rows),1):.1f}%)"]
    if not fired:
        return "\n".join(lines)
    lines.append(f"  mean detection confidence {np.mean([r['conf'] for r in fired]):.3f}")
    lines.append(f"  {'property':10s} {'median':>8s} {'p90':>8s} {'max':>8s}")
    for key, label in PROPS:
        v = np.array([r[key] for r in fired if key in r])
        if len(v) == 0:
            continue
        lines.append(f"  {label:10s} {np.median(v):7.2f}° {np.percentile(v,90):7.2f}° "
                     f"{v.max():7.2f}°")
    return "\n".join(lines)


def summarise(rows) -> dict:
    fired = [r for r in rows if r["fired"]]
    out = dict(n=len(rows), n_fired=len(fired),
               fire_rate=len(fired) / max(len(rows), 1),
               mean_conf=float(np.mean([r["conf"] for r in fired])) if fired else 0.0,
               props={})
    for key, _ in PROPS:
        v = np.array([r[key] for r in fired if key in r])
        if len(v):
            out["props"][key] = dict(median=float(np.median(v)),
                                     p90=float(np.percentile(v, 90)),
                                     max=float(v.max()))
    return out


def main(a):
    root = Path(a.root)
    items = load_index(root)
    rng = np.random.default_rng(0)
    if a.n and a.n < len(items):
        items = [items[i] for i in rng.choice(len(items), a.n, replace=False)]

    det = load_detector(Path(a.yolo), Path(a.repo))
    model, spec = load_landmark_model(Path(a.ckpt))
    rows = run(items, det, model, spec)

    print(_table(rows, "FETAL_PLANES_DB / Fetal thorax — external, unlabelled"))
    if a.by_machine:
        by = defaultdict(list)
        for r in rows:
            by[r["machine"]].append(r)
        for machine in sorted(by, key=lambda m: -len(by[m])):
            print()
            print(_table(by[machine], f"  by machine: {machine}"))

    if a.json:
        import json
        by = defaultdict(list)
        for r in rows:
            by[r["machine"]].append(r)
        Path(a.json).write_text(json.dumps(
            dict(overall=summarise(rows),
                 by_machine={m: summarise(v) for m, v in by.items()}), indent=1))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/FETAL_PLANES")
    p.add_argument("--yolo", default="runs/yolo/focus/weights/best.pt")
    p.add_argument("--repo", default="yolov5")
    p.add_argument("--ckpt", default="runs/landmarks/landmarks_best.pt")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--by-machine", action="store_true")
    p.add_argument("--json", default="runs/external.json")
    main(p.parse_args())
