"""Landmark dataset for cardiac orientation.

Stage 2 of the pipeline.  Given a crop around a detected heart, regress four
landmarks — the endpoints of the cardiac ellipse's major and minor axes — and
recover the orientation from them.

Why landmarks instead of regressing the angle directly:

* the output is *inspectable*.  A clinician can look at four points and say they
  are wrong; nobody can audit a scalar.
* it degrades gracefully — per-landmark confidence gives an abstention signal.
* both axes vote, and their disagreement is a label-free confidence estimate.
* it matches how the quantity is measured by hand, so the comparison against a
  human reader is like-for-like.

Landmark order is fixed: ``[major+, major-, minor+, minor-]``.  Because the heart
axis is *axial*, the two major endpoints are interchangeable; the canonical order
below resolves that ambiguity deterministically so the network is never punished
for a labelling convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .focus import Ellipse, Sample

K = 4  # number of landmarks


def canonical_landmarks(e: Ellipse) -> np.ndarray:
    """Axis endpoints in a deterministic order.

    The major axis is oriented so that its direction has a non-negative x
    component (and, on ties, non-negative y); the minor axis is then fixed by
    a +90 deg rotation from it.  Without this, two annotations of the same
    ellipse could differ by a swap and the loss would be discontinuous.
    """
    t = math.radians(e.theta)
    u = np.array([math.cos(t), math.sin(t)])
    if u[0] < 0 or (abs(u[0]) < 1e-9 and u[1] < 0):
        u = -u
    v = np.array([-u[1], u[0]])
    c = e.center
    return np.stack([c + e.a * u, c - e.a * u, c + e.b * v, c - e.b * v])


@dataclass
class CropSpec:
    size: int = 192
    margin: float = 0.35  # fraction of the box side added on every side


def crop_box(e: Ellipse, spec: CropSpec) -> tuple[float, float, float]:
    """Square crop around the heart: returns (x0, y0, side) in image pixels."""
    r = max(e.a, e.b) * (1.0 + spec.margin)
    return e.cx - r, e.cy - r, 2.0 * r


def affine_crop_matrix(
    cx: float, cy: float, r: float, size: int, rot_deg: float = 0.0
) -> np.ndarray:
    """2x3 affine mapping source pixels to a ``size``x``size`` crop.

    Rotation about (cx, cy) and the crop are composed into a single warp, so the
    image and the landmarks are transformed by exactly the same matrix and cannot
    drift apart through a convention mistake.  Out-of-image regions are filled
    with zeros, which is what ultrasound background is anyway.
    """
    M_rot = cv2.getRotationMatrix2D((float(cx), float(cy)), float(rot_deg), 1.0)
    s = size / (2.0 * r)
    M_crop = np.array([[s, 0.0, -s * (cx - r)], [0.0, s, -s * (cy - r)]], np.float64)
    A = np.vstack([M_rot, [0, 0, 1]])
    B = np.vstack([M_crop, [0, 0, 1]])
    return (B @ A)[:2]


def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    return pts @ M[:, :2].T + M[:, 2]


def axis_angle_under(M: np.ndarray, angle_deg: float) -> float:
    """How an axial angle transforms under the linear part of ``M``."""
    t = math.radians(angle_deg)
    v = M[:, :2] @ np.array([math.cos(t), math.sin(t)])
    return math.degrees(math.atan2(v[1], v[0])) % 180.0


def make_example(
    s: Sample,
    spec: CropSpec,
    jitter: np.random.Generator | None = None,
    rot_deg: float = 0.0,
    gain: tuple[float, float] = (1.0, 0.0),
    flip: bool = False,
) -> dict:
    """Build one example: crop, landmarks in crop coordinates, ground-truth angle.

    Augmentations are the physically meaningful ones for ultrasound: rotation
    (fetal lie is arbitrary), left-right flip (a valid lie), scale (depth
    setting) and gain/brightness (machine and operator).  No hue — the images are
    grayscale.  No vertical flip — that would swap near and far field, which no
    probe does, and it would teach the model an artifact that never occurs.
    """
    e = s.cardiac
    img = np.asarray(Image.open(s.image_path).convert("L"), np.float32) / 255.0
    pts = canonical_landmarks(e)

    cx, cy = e.cx, e.cy
    r = max(e.a, e.b) * (1.0 + spec.margin)
    if jitter is not None:
        cx += float(jitter.normal(0, 0.06 * r))
        cy += float(jitter.normal(0, 0.06 * r))
        r *= float(np.exp(jitter.normal(0, 0.12)))

    M = affine_crop_matrix(cx, cy, r, spec.size, rot_deg)
    arr = cv2.warpAffine(img, M, (spec.size, spec.size), flags=cv2.INTER_LINEAR, borderValue=0.0)
    local = apply_affine(M, pts)

    if flip:
        arr = np.ascontiguousarray(arr[:, ::-1])
        local[:, 0] = spec.size - 1 - local[:, 0]
        local = local[[0, 1, 3, 2]]  # the minor endpoints swap under mirroring
        M = np.array([[-1.0, 0.0, spec.size - 1.0], [0.0, 1.0, 0.0]]) @ np.vstack([M, [0, 0, 1]])

    a_gain, b_gain = gain
    if a_gain != 1.0 or b_gain != 0.0:
        arr = np.clip(arr * a_gain + b_gain, 0.0, 1.0)

    return dict(
        image=arr.astype(np.float32),
        landmarks=local.astype(np.float32),
        stem=s.stem,
        matrix=M,
        gt_angle=axis_angle_under(M, e.theta),
        gt_angle_image=e.theta,
        anisotropy=e.anisotropy,
    )
