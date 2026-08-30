"""Agreement statistics, and the label-free test machinery itself."""

import numpy as np
import pytest

from fho import geometry as G
from fho.evaluate import bland_altman, bootstrap_ci, icc21
from fho.metamorphic import expected_angle


def test_bland_altman_recovers_a_known_offset():
    ref = np.arange(0.0, 170.0, 5.0)
    pred = (ref + 3.0) % 180.0
    ba = bland_altman(pred, ref)
    assert ba["bias_deg"] == pytest.approx(3.0, abs=1e-6)
    assert ba["sd_deg"] < 1e-6
    assert ba["n"] == len(ref)


def test_bland_altman_handles_the_180_degree_wrap():
    """179 vs 1 is a 2-degree difference, not 178."""
    ba = bland_altman(np.array([179.0, 1.0]), np.array([1.0, 179.0]))
    assert abs(ba["bias_deg"]) < 1e-9
    assert ba["sd_deg"] == pytest.approx(2.0 * np.sqrt(2), abs=1e-6)


def test_icc_is_one_for_identical_measurements():
    a = np.array([10.0, 40.0, 75.0, 120.0, 160.0])
    assert icc21(a, a) == pytest.approx(1.0, abs=1e-9)


def test_icc_falls_when_agreement_falls():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 180, 200)
    good = icc21(a, a + rng.normal(0, 2, 200))
    poor = icc21(a, a + rng.normal(0, 40, 200))
    assert good > poor
    assert good > 0.9


def test_bootstrap_interval_brackets_the_statistic():
    rng = np.random.default_rng(0)
    v = rng.gamma(2.0, 3.0, 400)
    lo, hi = bootstrap_ci(v, n_boot=400)
    assert lo <= np.median(v) <= hi


@pytest.mark.parametrize("delta", [-30.0, -15.0, 15.0, 30.0])
def test_metamorphic_expectation_comes_from_the_matrix(delta):
    """The expectation must be derived, not assumed.

    An earlier version of this suite asserted ``base + delta`` and reported
    errors of exactly twice delta on every image — the signature of a flipped
    sign in the *test*.  Deriving it from the warp matrix removes the convention.
    """
    import math

    from fho.landmarks import affine_crop_matrix

    M0 = affine_crop_matrix(100.0, 100.0, 50.0, 192, 0.0)
    M1 = affine_crop_matrix(100.0, 100.0, 50.0, 192, delta)
    base = 20.0
    got = expected_angle(base, M0, M1)

    t = math.radians(delta)
    R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
    u = R @ np.array([math.cos(math.radians(base)), math.sin(math.radians(base))])
    want = math.degrees(math.atan2(u[1], u[0])) % 180.0
    assert G.angdiff_axial(got, want) < 1e-6


def test_expected_angle_is_identity_for_the_same_matrix():
    from fho.landmarks import affine_crop_matrix

    M = affine_crop_matrix(80.0, 60.0, 40.0, 192, 12.0)
    assert G.angdiff_axial(expected_angle(57.0, M, M), 57.0) < 1e-9
