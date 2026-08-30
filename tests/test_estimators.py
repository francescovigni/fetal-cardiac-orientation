"""The training-free estimators, and the failure modes they do and do not survive."""

import numpy as np
import pytest

from fho import geometry as G
from fho.no_training import bite, blob, boundary_noise, cleanup, dice, dilate, erode, estimate


def test_moments_recover_a_known_ellipse(ellipse_mask):
    m, theta = ellipse_mask()
    assert G.angdiff_axial(G.mask_axis(m)["angle"], theta) < 1.0


@pytest.mark.parametrize("k", [2, 5, 9])
def test_moments_survive_symmetric_erosion_and_dilation(ellipse_mask, k):
    """The headline property: symmetric segmentation error is nearly free."""
    m, theta = ellipse_mask()
    for fn in (erode, dilate):
        got = G.mask_axis(fn((m > 0).astype(np.uint8), k))["angle"]
        assert G.angdiff_axial(got, theta) < 2.0


def test_cleanup_removes_a_detached_blob(ellipse_mask):
    """A *detached* spurious component is removed and the angle is restored."""
    import cv2

    m, theta = ellipse_mask(a=60.0, b=25.0)
    m = (m > 0).astype(np.uint8)
    # place the blob *perpendicular* to the major axis, where spurious mass
    # rotates the estimate most; a blob along the axis barely moves it
    import math

    n = math.radians(theta + 90.0)
    cx, cy = 200 + 120 * math.cos(n), 150 + 120 * math.sin(n)
    corrupted = m.copy()
    cv2.circle(corrupted, (int(cx), int(cy)), 24, 1, -1)
    assert dice(corrupted, m) < 1.0

    e = estimate(corrupted)
    raw = G.angdiff_axial(e["moments"], theta)
    clean = G.angdiff_axial(e["cleaned"], theta)
    assert raw > 5.0, "the blob should have moved the raw estimate"
    assert clean < 2.0, "cleanup should have restored it"


def test_attached_leakage_is_not_removed(ellipse_mask):
    """The measured limit: spurious mass touching the object survives cleanup.

    Connected components cannot separate what is connected, which is why the
    README distinguishes detached blobs from tissue leaking across a contiguous
    boundary — the second is the failure a real segmenter actually produces.
    """
    m, theta = ellipse_mask(a=60.0, b=25.0)
    m = (m > 0).astype(np.uint8)
    corrupted = blob(m, 0.5)  # placed adjacent, so it touches
    e = estimate(corrupted)
    assert G.angdiff_axial(e["cleaned"], theta) > 2.0


def test_cleanup_keeps_only_the_largest_component(ellipse_mask):
    import cv2

    m, _ = ellipse_mask()
    m = (m > 0).astype(np.uint8)
    far = m.copy()
    cv2.circle(far, (20, 20), 12, 1, -1)
    n_before = cv2.connectedComponents(far)[0]
    n_after = cv2.connectedComponents(cleanup(far))[0]
    assert n_before == 3 and n_after == 2


def test_cleanup_rescues_a_ragged_contour(ellipse_mask):
    """40 degrees to under 5 is the measured effect; assert the direction strongly."""
    m, theta = ellipse_mask()
    m = (m > 0).astype(np.uint8)
    rng = np.random.default_rng(0)
    corrupted = boundary_noise(m, 1.2, rng)
    e = estimate(corrupted)
    assert G.angdiff_axial(e["cleaned"], theta) <= G.angdiff_axial(e["moments"], theta) + 1e-6


def test_asymmetric_mass_is_not_rescued(ellipse_mask):
    """The honest negative: a missing chunk survives the cleanup."""
    m, theta = ellipse_mask()
    m = (m > 0).astype(np.uint8)
    corrupted = bite(m, 0.45)
    e = estimate(corrupted)
    assert G.angdiff_axial(e["cleaned"], theta) > 2.0


def test_minimum_area_rectangle_is_bimodal_on_an_ellipse(ellipse_mask):
    """It is exact for area and picks one of two optima 90 degrees apart."""
    errs = []
    for theta in (10.0, 25.0, 40.0, 55.0, 70.0, 100.0, 130.0, 160.0):
        m, _ = ellipse_mask(a=60.0, b=45.0, theta=theta)
        ys, xs = np.nonzero(m)
        got = G.minarea_axis(np.stack([xs, ys], 1).astype(float))["angle"]
        errs.append(G.angdiff_axial(got, theta))
    errs = np.array(errs)
    assert (errs > 45).any(), "expected at least one ~90-degree flip"
    assert (errs < 5).any(), "expected at least one correct alignment"


def test_dice_is_one_for_an_identical_mask(ellipse_mask):
    m, _ = ellipse_mask()
    assert dice(m, m) == pytest.approx(1.0)


def test_pca_standard_error_grows_as_the_shape_rounds(ellipse_mask):
    thin, _ = ellipse_mask(a=80.0, b=15.0)
    round_, _ = ellipse_mask(a=80.0, b=78.0)
    assert G.mask_axis(round_)["angle_se_deg"] > 5 * G.mask_axis(thin)["angle_se_deg"]
