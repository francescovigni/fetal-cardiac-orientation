"""Crop geometry: the image and the landmarks must transform together."""

import math

import numpy as np
import pytest

from fho import geometry as G
from fho.focus import load_split
from fho.landmarks import (
    CropSpec,
    affine_crop_matrix,
    apply_affine,
    axis_angle_under,
    canonical_landmarks,
    make_example,
)


def test_canonical_ordering_is_deterministic_and_axis_preserving(fake_focus):
    for s in load_split(fake_focus, "train"):
        pts = canonical_landmarks(s.cardiac)
        assert pts.shape == (4, 2)
        assert G.angdiff_axial(G.axis_from_landmarks(pts)["angle"], s.cardiac.theta) < 1e-6
        u = pts[0] - pts[1]
        assert u[0] > 0 or (abs(u[0]) < 1e-9 and u[1] >= 0)


def test_crop_matrix_maps_the_centre_to_the_crop_centre():
    M = affine_crop_matrix(100.0, 80.0, 50.0, 192)
    c = apply_affine(M, np.array([[100.0, 80.0]]))[0]
    assert c == pytest.approx([96.0, 96.0], abs=1e-6)


@pytest.mark.parametrize("rot", [0.0, 17.0, -40.0, 123.0])
def test_image_and_landmarks_stay_consistent_under_rotation(fake_focus, rot):
    """The whole point of composing rotation and crop into one matrix."""
    s = load_split(fake_focus, "train")[0]
    ex = make_example(s, CropSpec(), rot_deg=rot)
    from_landmarks = G.axis_from_landmarks(ex["landmarks"])["angle"]
    assert G.angdiff_axial(from_landmarks, ex["gt_angle"]) < 1e-4


def test_mirroring_reflects_the_axis(fake_focus):
    s = load_split(fake_focus, "train")[0]
    base = make_example(s, CropSpec())
    flip = make_example(s, CropSpec(), flip=True)
    assert G.angdiff_axial(flip["gt_angle"], -base["gt_angle"]) < 1e-4
    assert (
        G.angdiff_axial(G.axis_from_landmarks(flip["landmarks"])["angle"], flip["gt_angle"]) < 1e-4
    )


def test_gain_does_not_move_the_landmarks(fake_focus):
    s = load_split(fake_focus, "train")[0]
    a = make_example(s, CropSpec())
    b = make_example(s, CropSpec(), gain=(1.5, -0.1))
    assert np.allclose(a["landmarks"], b["landmarks"])
    assert not np.allclose(a["image"], b["image"])


def test_crop_survives_a_heart_at_the_image_border(fake_focus):
    """Out-of-image regions must zero-fill rather than raise."""
    from fho.focus import Ellipse

    s = load_split(fake_focus, "train")[0]
    edge = Ellipse(5.0, 5.0, 60.0, 30.0, 20.0, "cardiac")
    object.__setattr__(s, "ellipses", {**s.ellipses, "cardiac": edge})
    ex = make_example(s, CropSpec())
    assert ex["image"].shape == (192, 192)
    assert np.isfinite(ex["image"]).all()


def test_axis_angle_under_matches_an_explicit_rotation():
    t = math.radians(25.0)
    M = np.array([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0]])
    assert G.angdiff_axial(axis_angle_under(M, 10.0), 35.0) < 1e-9
