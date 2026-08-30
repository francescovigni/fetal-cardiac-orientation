"""End-to-end inference: YOLOv5 heart detection -> crop -> landmarks -> angle.

    python -m fho.predict --image path/to/scan.png \
        --yolo runs/yolo/focus/weights/best.pt --ckpt runs/landmarks/landmarks_best.pt

Output is a dict with the heart box, the four landmarks, the axial angle, and the
confidence signals.  ``assessable`` is False when the estimate should not be
reported at all — see README §"Abstention".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import geometry as G
from .landmarks import CropSpec, affine_crop_matrix

# An orientation is not reported when the shape is too round for an axis to be
# defined, or when the two internal estimates disagree.  Both thresholds are
# tuned on the validation split by evaluate.py's risk-coverage table, not guessed.
ROUNDNESS_LIMIT = 0.93  # predicted b/a above this -> axis ill-defined
DISAGREEMENT_LIMIT = 12.0  # deg, between the major-axis and minor-axis votes
HEAD_DISAGREEMENT_LIMIT = 15.0  # deg, between the coordinate head and the axis head


def load_landmark_model(ckpt: Path):
    import torch

    from .landmarks import K
    from .model import LandmarkNet

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = LandmarkNet(K, state.get("width", 32))
    model.load_state_dict(state["model"])
    model.eval()
    return model, CropSpec(**state["spec"])


def detect_heart(
    yolo_weights: Path, image_path: Path, conf: float = 0.25, repo: Path = Path("yolov5")
):
    """Run YOLOv5 and return the highest-confidence 'cardiac' box, or None.

    Loaded from the local checkout (``source="local"``) so inference works
    offline and pins the exact code that trained the weights.
    """
    import torch

    model = torch.hub.load(
        str(repo), "custom", path=str(yolo_weights), source="local", verbose=False
    )
    model.conf = conf
    det = model(str(image_path)).xyxy[0].cpu().numpy()
    cardiac = det[det[:, 5] == 0]
    if len(cardiac) == 0:
        return None
    x0, y0, x1, y1 = cardiac[cardiac[:, 4].argmax()][:4]
    return float(x0), float(y0), float(x1), float(y1)


def orientation_from_box(
    model, spec: CropSpec, image: np.ndarray, box: tuple[float, float, float, float]
) -> dict:
    import torch

    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = max(x1 - x0, y1 - y0) / 2 * (1.0 + spec.margin)
    M = affine_crop_matrix(cx, cy, r, spec.size)
    crop = cv2.warpAffine(image, M, (spec.size, spec.size), flags=cv2.INTER_LINEAR, borderValue=0.0)

    with torch.no_grad():
        axis, coords = model(torch.from_numpy(crop)[None, None].float())
    pts_crop = coords[0].numpy() * 4.0
    d = G.axis_from_landmarks(pts_crop)
    direct = G.decode_axial(axis[0].numpy())

    Minv = cv2.invertAffineTransform(M)
    pts_img = pts_crop @ Minv[:, :2].T + Minv[:, 2]
    angle_img = G.axis_from_landmarks(pts_img)["angle"]

    head_gap = float(G.angdiff_axial(d["angle"], direct))
    assessable = (
        d["anisotropy"] < ROUNDNESS_LIMIT
        and d["disagreement"] < DISAGREEMENT_LIMIT
        and head_gap < HEAD_DISAGREEMENT_LIMIT
    )
    return dict(
        box=[x0, y0, x1, y1],
        landmarks=pts_img.tolist(),
        angle_deg=float(angle_img),
        angle_direct_head_deg=float(direct),
        roundness=float(d["anisotropy"]),
        axis_disagreement_deg=float(d["disagreement"]),
        head_disagreement_deg=head_gap,
        assessable=bool(assessable),
    )


def main(a):
    img = np.asarray(cv2.imread(str(a.image), cv2.IMREAD_GRAYSCALE), np.float32) / 255.0
    model, spec = load_landmark_model(Path(a.ckpt))
    box = detect_heart(Path(a.yolo), Path(a.image), a.conf, Path(a.repo)) if a.yolo else None
    if box is None:
        h, w = img.shape
        box = (w * 0.25, h * 0.25, w * 0.75, h * 0.75)
        print("# no detection — falling back to a centre crop", flush=True)
    print(json.dumps(orientation_from_box(model, spec, img, box), indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--yolo", default="runs/yolo/focus/weights/best.pt")
    p.add_argument("--ckpt", default="runs/landmarks/landmarks_best.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--repo", default="yolov5")
    main(p.parse_args())
