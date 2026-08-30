"""Network shapes, and the invariances the loss is supposed to have."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fho.model import LandmarkNet, Loss, axial_angle_from_coords  # noqa: E402


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return LandmarkNet(k=4, width=8)


def test_forward_shapes(net):
    x = torch.randn(3, 1, 192, 192)
    axis, coords = net(x)
    assert axis.shape == (3, 2)
    assert coords.shape == (3, 4, 2)
    assert torch.allclose(axis.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_coordinates_stay_inside_the_crop(net):
    _, coords = net(torch.randn(5, 1, 192, 192))
    assert (coords >= 0).all() and (coords <= 192 / 4).all()


def test_axis_from_coords_is_invariant_to_the_180_degree_swap():
    """The reason the angular loss needs no permutation handling."""
    c = torch.tensor([[[10.0, 0.0], [-10.0, 0.0], [0.0, 4.0], [0.0, -4.0]]])
    swapped = c[:, [1, 0, 3, 2]]
    assert torch.allclose(axial_angle_from_coords(c), axial_angle_from_coords(swapped), atol=1e-6)


def test_axis_from_coords_recovers_a_known_angle():
    import math

    for deg in (0.0, 30.0, 95.0, 170.0):
        t = math.radians(deg)
        u = torch.tensor([math.cos(t), math.sin(t)]) * 10
        v = torch.tensor([-math.sin(t), math.cos(t)]) * 4
        c = torch.stack([u, -u, v, -v])[None]
        z = axial_angle_from_coords(c)[0].numpy()
        got = np.degrees(np.arctan2(z[0], z[1])) / 2 % 180.0
        assert min(abs(got - deg), 180 - abs(got - deg)) < 1e-3


def test_loss_is_invariant_to_the_endpoint_swap():
    """Two labellings of the same ellipse must cost the same."""
    torch.manual_seed(1)
    crit = Loss()
    coords = torch.randn(4, 4, 2) * 10 + 24
    target = torch.randn(4, 4, 2) * 10 + 24
    axis = torch.nn.functional.normalize(torch.randn(4, 2), dim=-1)  # same axis both times
    a, _ = crit(axis, coords, target)
    b, _ = crit(axis, coords, target[:, [1, 0, 3, 2]])
    assert torch.allclose(a, b, atol=1e-5)


def test_loss_is_zero_for_a_perfect_prediction():
    crit = Loss()
    target = torch.tensor([[[34.0, 24.0], [14.0, 24.0], [24.0, 28.0], [24.0, 20.0]]])
    axis = axial_angle_from_coords(target)
    total, parts = crit(axis, target, target)
    assert float(total) < 1e-5
    assert parts["xy"] < 1e-6 and parts["ang"] < 1e-6 and parts["dir"] < 1e-6


def test_gradients_reach_every_parameter(net):
    crit = Loss()
    x = torch.randn(2, 1, 192, 192)
    target = torch.rand(2, 4, 2) * 40
    axis, coords = net(x)
    total, _ = crit(axis, coords, target)
    total.backward()
    missing = [
        n
        for n, p in net.named_parameters()
        if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))
    ]
    assert not missing, f"no gradient reached: {missing[:5]}"
