"""Angles, circular statistics, and the classical orientation baselines.

Everything here works in **image coordinates** (x right, y down) and treats heart
orientation as *axial* data: an axis is defined modulo 180 deg, not 360.  Direction
(apex left vs right) is situs and cannot be recovered from the heart alone — see
README §"Sign and situs".
"""
from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------- #
# axial angle helpers                                                          #
# --------------------------------------------------------------------------- #

def wrap180(a: np.ndarray | float) -> np.ndarray | float:
    """Map an axial angle into [0, 180)."""
    return np.mod(a, 180.0)


def angdiff_axial(a, b):
    """Smallest absolute difference between two axial angles, in [0, 90]."""
    d = np.abs(np.mod(np.asarray(a, float) - np.asarray(b, float), 180.0))
    return np.minimum(d, 180.0 - d)


def signed_angdiff_axial(a, b):
    """Signed difference in (-90, 90]:  a - b, taken the short way round."""
    d = np.mod(np.asarray(a, float) - np.asarray(b, float) + 90.0, 180.0) - 90.0
    return d


def circ_mean_axial(angles_deg) -> float:
    """Circular mean of axial angles (the 'double the angle' trick)."""
    z = np.exp(2j * np.radians(np.asarray(angles_deg, float)))
    return float(np.degrees(np.angle(z.mean())) / 2.0) % 180.0


def circ_sd_axial(angles_deg) -> float:
    """Circular standard deviation of axial angles, in degrees."""
    z = np.exp(2j * np.radians(np.asarray(angles_deg, float)))
    R = float(np.abs(z.mean()))
    R = min(max(R, 1e-12), 1.0 - 1e-15)
    return float(np.degrees(math.sqrt(-2.0 * math.log(R))) / 2.0)


def encode_axial(deg):
    """Axial angle -> (sin 2t, cos 2t).  Continuous across the 0/180 wrap."""
    t = np.radians(np.asarray(deg, float)) * 2.0
    return np.stack([np.sin(t), np.cos(t)], -1)


def decode_axial(sc):
    """(sin 2t, cos 2t) -> axial angle in [0, 180)."""
    sc = np.asarray(sc, float)
    return np.degrees(np.arctan2(sc[..., 0], sc[..., 1])) / 2.0 % 180.0


# --------------------------------------------------------------------------- #
# orientation from a point set / mask                                          #
# --------------------------------------------------------------------------- #

def pca_axis(points: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Principal-axis orientation of a weighted 2-D point set.

    Returns angle (deg, axial), the two eigenvalues, the anisotropy lam2/lam1 and
    the first-order angular standard error implied by the eigengap:

        Var(theta) ~ (1/n) * lam1*lam2 / (lam1 - lam2)^2

    That standard error is the abstention signal: when the shape is round the gap
    vanishes and the angle is noise.
    """
    P = np.asarray(points, float)
    n = len(P)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    mu = (w[:, None] * P).sum(0)
    X = P - mu
    C = (X * w[:, None]).T @ X
    lam, R = np.linalg.eigh(C)
    order = lam.argsort()[::-1]
    lam, R = lam[order], R[:, order]
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    gap = max(lam[0] - lam[1], 1e-12)
    se = math.degrees(math.sqrt(lam[0] * lam[1] / gap ** 2 / max(n, 1)))
    Y = X @ R
    lo, hi = Y.min(0), Y.max(0)
    return dict(
        angle=float(math.degrees(math.atan2(R[1, 0], R[0, 0])) % 180.0),
        center=mu + R @ ((lo + hi) / 2),
        size=hi - lo,
        eigvals=lam,
        anisotropy=float(lam[1] / lam[0]) if lam[0] > 0 else 1.0,
        angle_se_deg=float(se),
    )


def mask_axis(mask: np.ndarray, spacing: tuple[float, float] = (1.0, 1.0)) -> dict:
    """PCA axis of a binary or soft mask, via probability-weighted image moments.

    ``spacing`` is (mm per px in x, mm per px in y) and must be applied *before*
    the moments: anisotropic pixels skew the axis by the pixel aspect ratio.
    """
    m = np.asarray(mask, float)
    ys, xs = np.nonzero(m > 0)
    if len(xs) < 3:
        return dict(angle=float("nan"), anisotropy=1.0, angle_se_deg=float("inf"))
    P = np.stack([xs * spacing[0], ys * spacing[1]], 1).astype(float)
    return pca_axis(P, m[ys, xs])


def _convex_hull(P: np.ndarray) -> np.ndarray:
    P = np.unique(np.asarray(P, float), axis=0)
    P = P[np.lexsort((P[:, 1], P[:, 0]))]
    if len(P) <= 2:
        return P

    def half(pts):
        st: list[np.ndarray] = []
        for p in pts:
            while len(st) >= 2:
                a, b = st[-2], st[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 0:
                    break
                st.pop()
            st.append(p)
        return st

    return np.array(half(P)[:-1] + half(P[::-1])[:-1])


def minarea_axis(points: np.ndarray) -> dict:
    """Exact minimum-area enclosing rectangle (rotating calipers over hull edges).

    Optimal for *area*; a poor and unstable *axis* estimator on near-round or
    ragged shapes, because the argmin jumps between hull edges.  Kept as the
    other classical baseline so the comparison is on the record.
    """
    P = np.asarray(points, float)
    H = _convex_hull(P)
    best = None
    for i in range(len(H)):
        e = H[(i + 1) % len(H)] - H[i]
        n = float(np.linalg.norm(e))
        if n < 1e-12:
            continue
        u = e / n
        R = np.stack([u, np.array([-u[1], u[0]])], 1)
        Y = P @ R
        lo, hi = Y.min(0), Y.max(0)
        area = float(np.prod(hi - lo))
        if best is None or area < best["area"]:
            best = dict(area=area,
                        angle=float(math.degrees(math.atan2(u[1], u[0])) % 180.0),
                        size=hi - lo)
    return best or dict(area=float("nan"), angle=float("nan"))


# --------------------------------------------------------------------------- #
# landmark -> orientation                                                      #
# --------------------------------------------------------------------------- #

def axis_from_landmarks(pts: np.ndarray) -> dict:
    """Heart long axis from the four cardiac ellipse axis endpoints.

    ``pts`` is (4, 2) ordered [major+, major-, minor+, minor-].  Both axes vote:
    the major endpoints give the axis directly, the minor endpoints give it
    rotated by 90 deg, and the two are combined with a circular mean.  Their
    disagreement is a free, label-free confidence signal.
    """
    pts = np.asarray(pts, float).reshape(4, 2)
    d_major = pts[0] - pts[1]
    d_minor = pts[2] - pts[3]
    a_major = math.degrees(math.atan2(d_major[1], d_major[0])) % 180.0
    a_minor = (math.degrees(math.atan2(d_minor[1], d_minor[0])) + 90.0) % 180.0
    len_major = float(np.linalg.norm(d_major))
    len_minor = float(np.linalg.norm(d_minor))
    angle = circ_mean_axial([a_major, a_minor])
    return dict(
        angle=angle,
        angle_major=a_major,
        angle_minor=a_minor,
        disagreement=float(angdiff_axial(a_major, a_minor)),
        center=pts[:2].mean(0),
        semi_major=len_major / 2,
        semi_minor=len_minor / 2,
        anisotropy=float(len_minor / len_major) if len_major > 0 else 1.0,
    )


def cardiac_axis(heart_angle_deg: float, ap_midline_deg: float) -> float:
    """Clinical cardiac axis = heart long axis relative to the thoracic AP midline.

    Both inputs are axial angles in degrees.  The result is signed, in (-90, 90].
    Sign is only meaningful once left/right is fixed — see README §"Sign and situs".
    """
    return float(signed_angdiff_axial(heart_angle_deg, ap_midline_deg))
