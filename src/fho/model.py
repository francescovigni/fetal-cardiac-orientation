"""Landmark-regression network for cardiac orientation.

**Why global regression and not heatmaps.**  The first version of this file used
per-landmark Gaussian heatmaps.  It does not work for these landmarks, and the
reason is worth keeping: the endpoints of the cardiac ellipse's axes have no
distinctive *local* appearance.  They are points on a smooth boundary, defined by
a global property of the shape (the extremum along a direction), so a receptive
field looking at a patch around one of them sees the same thing it sees a few
pixels along the contour.  Heatmap regression is the right tool for landmarks with
local evidence — an apex, a valve hinge, a vertebral body — and the wrong tool
here.  Measured: heatmaps plateaued at ~28 deg median error, barely better than
the 45 deg of chance for axial data.

So the landmarks are regressed globally from pooled features.  Two heads:

* ``coords`` — the four axis endpoints, the reported output, inspectable by a
  clinician;
* ``axis``   — the doubled-angle encoding predicted directly.

They are trained together and must agree.  Their disagreement at inference is a
label-free confidence signal, alongside the major/minor axis disagreement that
falls out of the four landmarks themselves.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class LandmarkNet(nn.Module):
    """(B, 1, S, S) -> coords (B, K, 2) in heatmap-stride units, axis (B, 2)."""

    def __init__(self, k: int = 4, width: int = 32, stride: int = 4):
        super().__init__()
        w = width
        self.k, self.stride = k, stride
        self.body = nn.Sequential(
            conv_bn(1, w, 2),
            conv_bn(w, w),  # /2
            conv_bn(w, 2 * w, 2),
            conv_bn(2 * w, 2 * w),  # /4
            conv_bn(2 * w, 4 * w, 2),
            conv_bn(4 * w, 4 * w),  # /8
            conv_bn(4 * w, 8 * w, 2),
            conv_bn(8 * w, 8 * w),  # /16
            conv_bn(8 * w, 8 * w, 2),
            conv_bn(8 * w, 8 * w),  # /32
        )
        # A 4x4 grid, not a 1x1 global average.  Global pooling is very nearly
        # orientation-invariant — it discards exactly the spatial layout that
        # encodes the angle.  Measured: GAP plateaued around 21 deg median error,
        # the 4x4 grid below converges to a fraction of that.
        # grid 3: with a 192 px crop the body outputs 6x6, and MPS only
        # implements adaptive pooling when the sizes divide evenly.
        self.pool = nn.AdaptiveAvgPool2d(3)
        self.trunk = nn.Sequential(
            nn.Linear(8 * w * 9, 256), nn.SiLU(inplace=True), nn.Dropout(0.1)
        )
        self.coord_head = nn.Linear(256, k * 2)
        self.axis_head = nn.Linear(256, 2)

    def forward(self, x):
        s = x.shape[-1]
        f = self.trunk(self.pool(self.body(x)).flatten(1))
        # coordinates are predicted normalised to [-1, 1] around the crop centre,
        # then mapped to heatmap-stride units so the rest of the code is unchanged
        c = torch.tanh(self.coord_head(f)).reshape(-1, self.k, 2)
        coords = (c + 1.0) * 0.5 * (s / self.stride)
        axis = F.normalize(self.axis_head(f), dim=-1, eps=1e-6)
        return axis, coords


def axial_angle_from_coords(coords: torch.Tensor) -> torch.Tensor:
    """(B, K, 2) -> (B, 2) unit vector (sin 2t, cos 2t) of the heart long axis.

    Both axes vote: the major endpoints directly, the minor endpoints rotated by
    90 deg — which in the doubled-angle representation is simply a negation.
    Doubling also makes the result invariant to the endpoint swap, which is why
    the angular term needs no permutation handling.
    """
    d_major = coords[:, 0] - coords[:, 1]
    d_minor = coords[:, 2] - coords[:, 3]
    t_major = torch.atan2(d_major[:, 1], d_major[:, 0]) * 2
    t_minor = torch.atan2(d_minor[:, 1], d_minor[:, 0]) * 2
    z = torch.stack([torch.sin(t_major), torch.cos(t_major)], -1) - torch.stack(
        [torch.sin(t_minor), torch.cos(t_minor)], -1
    )
    return F.normalize(z, dim=-1, eps=1e-6)


class Loss(nn.Module):
    """Swap-invariant coordinate loss + angular loss on both heads.

    An ellipse is invariant under a 180 deg rotation, which exchanges *both* pairs
    of axis endpoints at once.  Two labellings are therefore equally correct, and
    any fixed convention is discontinuous somewhere: the network would get
    contradictory targets for visually identical crops.  Scoring both assignments
    and keeping the better one per sample is the landmark analogue of the
    doubled-angle encoding used for the angle itself.
    """

    SWAPS = ([0, 1, 2, 3], [1, 0, 3, 2])

    def __init__(self, w_xy=1.0, w_ang=2.0, w_direct=1.0):
        super().__init__()
        self.w_xy, self.w_ang, self.w_direct = w_xy, w_ang, w_direct

    def forward(self, axis, coords, coords_t):
        per_swap = torch.stack(
            [(coords - coords_t[:, perm]).abs().mean((1, 2)) for perm in self.SWAPS], 0
        )
        l_xy = per_swap.min(0).values.mean()

        z_t = axial_angle_from_coords(coords_t)
        l_ang = (1.0 - (axial_angle_from_coords(coords) * z_t).sum(-1)).mean()
        l_dir = (1.0 - (axis * z_t).sum(-1)).mean()

        total = self.w_xy * l_xy + self.w_ang * l_ang + self.w_direct * l_dir
        return total, dict(
            xy=float(l_xy.detach()), ang=float(l_ang.detach()), dir=float(l_dir.detach())
        )
