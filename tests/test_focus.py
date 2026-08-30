"""Dataset parsing, annotation conventions, and the cross-validation check."""

import numpy as np
import pytest

from fho import geometry as G
from fho.focus import Ellipse, load_split, obb_angle, verify_consistency


def test_load_split_reads_all_annotation_formats(fake_focus):
    for split in ("train", "val", "test"):
        samples = load_split(fake_focus, split)
        assert len(samples) == 4
        s = samples[0]
        assert {"cardiac", "thorax"} <= set(s.ellipses)
        assert {"cardiac", "thorax"} <= set(s.obbs)
        assert s.obbs["cardiac"].shape == (4, 2)
        assert {"cardiac", "thorax"} <= set(s.mask_paths)
        assert s.image_path.exists()


def test_ellipse_and_oriented_box_agree(fake_focus):
    """The check that licenses using the ellipse angle as ground truth."""
    r = verify_consistency(load_split(fake_focus, "train"))
    assert r["n"] == 4
    assert r["center_px"]["max"] < 0.01
    assert r["semi_major_px"]["max"] < 0.01
    assert r["angle_deg"]["max"] < 0.01


def test_semi_major_is_normalised_when_b_exceeds_a(tmp_path):
    """b > a in the file must be swapped, with theta rotated by 90 degrees."""
    from fho.focus import _read_ellipse_file

    f = tmp_path / "x.txt"
    f.write_text("100 50 20 40 10 cardiac\n")
    e = _read_ellipse_file(f)["cardiac"]
    assert e.a == 40 and e.b == 20
    assert G.angdiff_axial(e.theta, 100.0) < 1e-9


def test_obb_angle_matches_the_long_edge():
    e = Ellipse(100.0, 50.0, 40.0, 20.0, 33.0, "cardiac")
    pts = e.axis_endpoints()
    corners = np.stack(
        [
            pts[0] + (pts[2] - e.center),
            pts[0] + (pts[3] - e.center),
            pts[1] + (pts[3] - e.center),
            pts[1] + (pts[2] - e.center),
        ]
    )
    assert G.angdiff_axial(obb_angle(corners), 33.0) < 1e-6


@pytest.mark.parametrize("theta", [0.0, 45.0, 90.0, 137.0])
def test_aabb_encloses_the_ellipse(theta):
    e = Ellipse(200.0, 150.0, 60.0, 25.0, theta, "cardiac")
    x0, y0, x1, y1 = e.aabb()
    for p in e.axis_endpoints():
        assert x0 - 1e-6 <= p[0] <= x1 + 1e-6
        assert y0 - 1e-6 <= p[1] <= y1 + 1e-6


def test_anisotropy_flags_a_round_shape():
    assert Ellipse(0, 0, 40.0, 10.0, 0.0, "c").anisotropy == pytest.approx(0.25)
    assert Ellipse(0, 0, 40.0, 39.0, 0.0, "c").anisotropy > 0.95
