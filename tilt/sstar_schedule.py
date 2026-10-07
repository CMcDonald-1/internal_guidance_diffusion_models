"""A time grid that is uniform in s*, the network's own input coordinate.

Why
---
`tilt/sampler.py::time_grid` builds its grid in t. But the network is
g_theta(v, s) with beta(s) = s/(1-s^2), and at SL time t the sampler queries

    eta = nu + X_t,   B = ||eta|| / sqrt(d),   s* = 2 / (sqrt(1/B^2 + 4) + 1/B)

so s* -- not t -- is the coordinate along which the drift actually varies.
Uniform / late / early are all schedules in the wrong variable.

Concretely, at tau=5, d=109, T=20:

    s* runs 0.402 -> 0.976        (beta 0.48 -> 20.5)

and the hard band beta in [0.5, 1.6] (write-up Section 2) occupies t in
[0.02, 1.11] -- 5.4% of the t-axis. Step allocation into that band, K=1000:

    late             28 steps   (2.8%)
    uniform          54 steps   (5.4%)
    early           204 steps   (20.4%)
    uniform in s*   559 steps   (55.9%)

`late` was designed for the argmax hardening near T. Under a reward tilt there
is a SECOND stiff region at t~0, where s* moves fastest and where lambda_eff
runs out to sqrt(d). `late` gives that region the coarsest steps in the whole
trajectory. Uniform-in-s* handles both ends automatically, because it is dense
in t exactly where s* is moving fast (first dt ~ 9e-4, last dt ~ 0.48).

This is a sampling-side change only -- no retraining.

Usage
-----
    from sstar_schedule import time_grid_sstar
    ts = time_grid_sstar(T=20.0, K=1000, h=h0, d=109)

then run the same Euler-Maruyama loop over `ts`. Or patch `time_grid` to
accept schedule="sstar" and pass h through from sl_sample.
"""
from __future__ import annotations

import math

import torch


def s_of_B(B: float) -> float:
    """Invert beta(s) = s/(1-s^2) = B, in the numerically stable form of Sec 8.4.

    s* = 2 / (sqrt(r^2+4) + r) with r = 1/B. Stable for all B >= 0:
    B -> 0 gives s* -> 0, B -> inf gives s* -> 1.
    """
    if B <= 0.0:
        return 0.0
    r = 1.0 / B
    return 2.0 / (math.sqrt(r * r + 4.0) + r)


def _B_of_t(t: float, nu_norm: float, d: int) -> float:
    """E||nu + X_t|| / sqrt(d) under X_t = tX + sqrt(t)Z, ||X||^2 = d.

    ||nu + X_t||^2 ~ ||nu||^2 + 2 t <nu, X> + d(t^2 + t). The cross term is
    taken at its aligned value t*||nu|| -- i.e. assuming the tilt succeeds and
    X agrees with nu in the tilted coordinate. That is the regime the schedule
    is for; if the tilt is weak the grid is merely slightly conservative.
    """
    return math.sqrt(nu_norm**2 + 2.0 * t * nu_norm + d * (t * t + t)) / math.sqrt(d)


def _t_of_B(B: float, nu_norm: float, d: int) -> float:
    """Inverse of _B_of_t: solve d t^2 + (d + 2 nu_norm) t + (nu^2 - d B^2) = 0."""
    a = float(d)
    b = float(d) + 2.0 * nu_norm
    c = nu_norm**2 - d * B * B
    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        return 0.0
    return max(0.0, (-b + math.sqrt(disc)) / (2.0 * a))


def grid_sstar_list(T: float, K: int, nu_norm: float, d: int) -> list[float]:
    """Pure-python core: the s*-uniform grid as a list of floats.

    Separated from the torch wrapper so the schedule maths can be tested
    without a torch install.
    """
    s0 = s_of_B(_B_of_t(0.0, nu_norm, d))
    sT = s_of_B(_B_of_t(T, nu_norm, d))
    ts = []
    for i in range(K + 1):
        s = s0 + (sT - s0) * i / K
        if s <= 0.0:
            ts.append(0.0)
        elif s >= 1.0:
            ts.append(T)
        else:
            ts.append(_t_of_B(s / (1.0 - s * s), nu_norm, d))
    # Pin the endpoints exactly -- the inversion is analytic but float-noisy.
    ts[0] = 0.0
    ts[-1] = T
    return ts


def time_grid_sstar(T: float, K: int, h: "torch.Tensor | float | None" = None,
                    d: int | None = None):
    """Grid 0 = t_0 < ... < t_K = T, equally spaced in s* rather than in t.

    h is the sampling tilt vector (tau * e_star, or the combined infill+reward
    nu); only its norm matters. h=None or 0 recovers the untilted case, where
    s* starts at 0.
    """
    if h is None:
        nu_norm = 0.0
    elif hasattr(h, "norm"):
        nu_norm = float(h.norm())
    else:
        nu_norm = float(h)
    if d is None:
        if hasattr(h, "shape"):
            d = h.shape[-1]
        else:
            raise ValueError("d must be given when h is not a tensor")
    return torch.tensor(grid_sstar_list(T, K, nu_norm, d), dtype=torch.float32)


# ------------------------------------------------------------------ patch form
PATCH = '''
# --- add to tilt/sampler.py::time_grid, and pass h through from sl_sample ---
def time_grid(T, K, schedule="uniform", h=None, d=None):
    s = torch.linspace(0.0, 1.0, K + 1)
    if schedule == "uniform":
        return T * s
    if schedule == "late":
        return T * (1.0 - (1.0 - s) ** 2)
    if schedule == "early":
        return T * s**2
    if schedule == "both":                       # dense at BOTH ends
        return T * (1.0 - torch.cos(math.pi * s)) / 2.0
    if schedule == "sstar":                      # uniform in the network's clock
        from .sstar_schedule import time_grid_sstar
        return time_grid_sstar(T, K, h=h, d=d)
    raise ValueError(f"unknown schedule {schedule!r}")

# in sl_sample, replace:
#     ts = time_grid(T, K, schedule)
# with:
#     ts = time_grid(T, K, schedule, h=h, d=d)
'''
