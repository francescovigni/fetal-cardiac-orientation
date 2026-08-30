"""Unit tests for the angle algebra.  Run with:  .venv/bin/python -m pytest -q"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fho import geometry as G
from fho.focus import Ellipse


def test_axial_difference_wraps():
    assert G.angdiff_axial(179.0, 1.0) == 2.0
    assert G.angdiff_axial(89.0, 91.0) == 2.0
    assert G.angdiff_axial(0.0, 90.0) == 90.0


def test_signed_difference_is_short_way():
    assert abs(G.signed_angdiff_axial(179.0, 1.0) - (-2.0)) < 1e-9
    assert abs(G.signed_angdiff_axial(1.0, 179.0) - 2.0) < 1e-9


def test_circular_mean_crosses_the_wrap():
    # linear mean would say 90; the circular mean must say 0
    assert G.angdiff_axial(G.circ_mean_axial([179.0, 1.0]), 0.0) < 1e-6


def test_encode_decode_roundtrip():
    a = np.arange(0.0, 180.0, 0.5)
    assert np.max(G.angdiff_axial(G.decode_axial(G.encode_axial(a)), a)) < 1e-9


def test_ellipse_endpoints_recover_the_angle():
    for theta in (0.0, 17.3, 89.9, 120.0, 179.0):
        e = Ellipse(100.0, 50.0, 40.0, 20.0, theta, "cardiac")
        got = G.axis_from_landmarks(e.axis_endpoints())["angle"]
        assert G.angdiff_axial(got, theta) < 1e-6


def test_pca_axis_on_a_known_cloud():
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 1, (4000, 2)) * [10.0, 2.0]
    t = math.radians(35.0)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    got = G.pca_axis(pts @ R.T)
    assert G.angdiff_axial(got["angle"], 35.0) < 1.0
    assert got["angle_se_deg"] < 0.5


def test_pca_standard_error_blows_up_when_round():
    rng = np.random.default_rng(0)
    elongated = G.pca_axis(rng.normal(0, 1, (2000, 2)) * [10.0, 2.0])
    round_ = G.pca_axis(rng.normal(0, 1, (2000, 2)) * [10.0, 9.8])
    assert round_["angle_se_deg"] > 10 * elongated["angle_se_deg"]


def test_minarea_is_never_larger_than_pca_box():
    rng = np.random.default_rng(1)
    pts = rng.normal(0, 1, (500, 2)) * [8.0, 3.0]
    mar = G.minarea_axis(pts)
    pca = G.pca_axis(pts)
    assert mar["area"] <= float(np.prod(pca["size"])) + 1e-6


def test_cardiac_axis_is_signed():
    assert abs(G.cardiac_axis(60.0, 15.0) - 45.0) < 1e-9
    assert abs(G.cardiac_axis(15.0, 60.0) + 45.0) < 1e-9
