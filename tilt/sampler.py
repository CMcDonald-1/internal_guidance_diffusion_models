"""Stochastic-localization sampler.

    dX_t = m(h + X_t) dt + dB_t,     X_0 = 0

where m is the tilt mean operator and h is the natural parameter of whatever
tilt you want to sample (h = 0 recovers pi itself; h = tau * e_{d+1} is the
reward tilt).  The drift IS an evaluation of the operator, so the trained
network plugs straight in via net.natural().

Passing drift_fn=target.natural_mean runs the same sampler with the EXACT
operator, which isolates discretisation error from network error -- run both
and difference them before blaming the network for a bad TV.

Design note, mirroring data.py: sl_sample returns the raw continuous X_T in
R^d, NOT a hard {-1,1}^d decision.  Rounding to hard spins is a separate,
swappable "decode" step (decode_drift, decode_sign, ...), and raw draws can be
persisted via save_samples/load_samples exactly like the training CSVs in
data.py, so decoding and TV measurement can be re-run against a fixed set of
raw draws without re-simulating the SDE.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import torch

SCHEDULES = ("uniform", "late", "early", "both", "sstar")


def time_grid(T: float, K: int, schedule: str = "uniform",
              h: torch.Tensor | None = None, d: int | None = None) -> torch.Tensor:
    """Grid 0 = t_0 < ... < t_K = T.

    'late' clusters steps near T, where the operator approaches a hard argmax
    and the drift's Lipschitz constant is largest.

    Under a reward tilt there is a SECOND stiff region, at t ~ 0: the drift is
    evaluated at h + X_t with X_0 = 0, so the first query direction is h/||h||
    itself, and the effective focusing level lambda_eff runs out to sqrt(d)
    before decaying as ||X_t|| grows. 'late' gives that region the coarsest
    steps in the whole trajectory. 'both' and 'sstar' address this:

      both   cosine grid, dt -> 0 at t=0 AND t=T. Cheap, tilt-agnostic.
      sstar  uniform in s*, the scalar the network actually takes as input
             (beta(s) = s/(1-s^2), s* from normalize_natural). Since the drift
             varies along s* rather than t, equal steps in s* equalise the
             change in drift per step. Requires h, and adapts to it.

    h and d are only used by 'sstar'; the other schedules ignore them.
    """
    s = torch.linspace(0.0, 1.0, K + 1)
    if schedule == "uniform":
        return T * s
    if schedule == "late":
        return T * (1.0 - (1.0 - s) ** 2)
    if schedule == "early":
        return T * s**2
    if schedule == "both":
        return T * (1.0 - torch.cos(math.pi * s)) / 2.0
    if schedule == "sstar":
        from .sstar_schedule import time_grid_sstar
        if d is None:
            raise ValueError("schedule='sstar' needs d (pass it from sl_sample)")
        return time_grid_sstar(T, K, h=h, d=d)
    raise ValueError(f"unknown schedule {schedule!r} (expected one of {SCHEDULES})")


@torch.no_grad()
def sl_sample(drift_fn, d: int, n: int, h: torch.Tensor | None = None,
              T: float = 20.0, K: int = 200, schedule: str = "late",
              seed: int = 0, return_path: bool = False,
              device: str | torch.device | None = None):
    """Euler-Maruyama on the SL SDE.  Returns the raw continuous X_T in R^d.

    No rounding happens here -- pair the result with a decode_* function (or
    persist it via save_samples and decode later) to get hard {-1,1}^d spins.

    device: where the trajectory lives. DEFAULT None = CPU, i.e. byte-for-byte
    the historical behaviour including the RNG stream, so every existing caller
    and every recorded number is unchanged. Pass 'cuda' to keep x, h and the
    noise on the GPU alongside the network.

    Why it matters: this loop runs K sequential forward passes of the operator,
    and with everything on CPU the network is evaluated on CPU too -- an earlier
    profile flagged this as "1000 sequential forward passes on CPU while the GPU sat
    idle". At d=151 that was merely slow; at d=481 with a 38M-parameter
    transformer it is the difference between seconds and an hour. Note the CUDA
    generator draws a DIFFERENT noise stream than the CPU one for the same seed,
    so a device='cuda' run is not bit-comparable to a device=None run -- it is a
    different draw from the same distribution, not a reproduction.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    g = torch.Generator(device=dev).manual_seed(seed)
    h = torch.zeros(d, device=dev) if h is None else h.to(dev)
    ts = time_grid(T, K, schedule, h=h.cpu(), d=d)
    x = torch.zeros(n, d, device=dev)
    path = []

    for k in range(K):
        dt = float(ts[k + 1] - ts[k])
        drift = drift_fn(h.unsqueeze(0) + x)
        noise = torch.randn(n, d, generator=g, device=dev) * (dt**0.5)
        x = x + dt * drift + noise
        if return_path:
            path.append((float(ts[k + 1]), x.clone()))

    return (x, path) if return_path else x


def decode_drift(x: torch.Tensor, drift_fn, h: torch.Tensor | None = None) -> torch.Tensor:
    """Round via the posterior-mean drift at x, lower-variance than decode_sign.

    This is the decode sl_sample used internally before the raw/decode split.
    """
    d = x.shape[1]
    h = torch.zeros(d) if h is None else h
    hard = torch.sign(drift_fn(h.unsqueeze(0) + x))
    hard[hard == 0] = 1.0
    return hard


def decode_sign(x: torch.Tensor) -> torch.Tensor:
    """Naive elementwise rounding of the raw draw itself."""
    hard = torch.sign(x)
    hard[hard == 0] = 1.0
    return hard


def save_samples(x: torch.Tensor, outpath: str | Path, meta: dict | None = None) -> Path:
    """Persist raw R^d draws to CSV, same x0..x{d-1} column convention as data.py."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    d = x.shape[1]
    cols = {f"x{i}": x[:, i].numpy() for i in range(d)}
    pd.DataFrame(cols).to_csv(outpath, index=False)
    if meta is not None:
        outpath.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return outpath


def load_samples(path: str | Path) -> torch.Tensor:
    """Load raw R^d draws previously written by save_samples."""
    path = Path(path)
    df = pd.read_csv(path)
    d = sum(1 for c in df.columns if c.startswith("x"))
    return torch.tensor(df[[f"x{i}" for i in range(d)]].to_numpy(), dtype=torch.float32)


def empirical_probs(samples: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """Histogram hard {-1,1}^d samples over the enumerated state space."""
    d = states.shape[1]
    powers = (2 ** torch.arange(d)).float()
    idx = (((samples > 0).float()) * powers).sum(1).long()
    counts = torch.bincount(idx, minlength=states.shape[0]).float()
    return counts / counts.sum()


def total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(0.5 * (p - q).abs().sum())
