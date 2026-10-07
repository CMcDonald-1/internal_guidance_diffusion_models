"""Stochastic-localization sampler for SelfCondTiltMLP -- a standalone parallel to
sampler.py. Reuses time_grid, save_samples, load_samples, empirical_probs,
total_variation from sampler.py unmodified (imported, not duplicated); only the
drift-evaluation step differs, since self-conditioning needs to carry an x_pred
state across Euler steps that a generic drift_fn(h) callable can't express.

    dX_t = m(h + X_t, x_pred_t) dt + dB_t,     X_0 = 0,   x_pred_0 = 0

Each step's own network output becomes next step's self-conditioning input (Bit
Diffusion's Algorithm 2 default strategy -- one forward pass per step, not the
2x-cost Algorithm 6 self-guidance variant).
"""
from __future__ import annotations

import warnings

import torch

from .sampler import time_grid


def _check_self_cond_mode(self_cond_mode: str) -> None:
    if self_cond_mode not in ("two_pass", "carry"):
        raise ValueError(
            f"self_cond_mode must be 'two_pass' or 'carry', got {self_cond_mode!r}")
    if self_cond_mode == "carry":
        warnings.warn(
            "self_cond_mode='carry' is non-Markov (see Appendix B.4 of "
            "the paper); it is kept because some reported results use it.", DeprecationWarning, stacklevel=3)


@torch.no_grad()
def sl_sample_selfcond(net, d: int, n: int, h: torch.Tensor | None = None,
                        T: float = 20.0, K: int = 200, schedule: str = "late",
                        seed: int = 0, return_path: bool = False,
                        device: str | torch.device | None = None,
                        self_cond_mode: str = "two_pass"):
    """Euler-Maruyama on the SL SDE with self-conditioning. Returns the raw
    continuous X_T in R^d (no rounding -- pair with decode_drift_selfcond).

    self_cond_mode (see Appendix B.4 of the paper):
      "two_pass" (DEFAULT): x_pred recomputed at the CURRENT state each step,
          so the drift is a pure function of h + X_t, the chain is Markov, and
          the path law is the SL law of net.self_cond_predict. 2 passes/step.
      "carry" (DEPRECATED): x_pred is the previous step's output (1 pass/step);
          non-Markov, sampled law is not the SL law of any fixed operator.
          Non-Markov (Appendix B.4); emits a warning.
    """
    _check_self_cond_mode(self_cond_mode)
    dev = torch.device(device) if device is not None else torch.device("cpu")
    g = torch.Generator(device=dev).manual_seed(seed)
    h = torch.zeros(d, device=dev) if h is None else h.to(dev)
    # h may be (d,) -- one tilt for the whole batch, the historical case -- or
    # (n, d), one per row. Infilling needs the latter: every sequence clamps a
    # different set of known bits. Broadcasting handles both once h is 2-D.
    hb = h if h.dim() == 2 else h.unsqueeze(0)
    ts = time_grid(T, K, schedule, h=(h if h.dim() == 1 else h[0]).cpu(), d=d)
    x = torch.zeros(n, d, device=dev)
    x_pred = torch.zeros(n, d, device=dev)
    path = []

    for k in range(K):
        dt = float(ts[k + 1] - ts[k])
        if self_cond_mode == "two_pass":
            x_pred = net.natural(hb + x, x_pred=None)  # fresh estimate at current state
        drift = net.natural(hb + x, x_pred=x_pred)
        if self_cond_mode == "carry":
            x_pred = drift  # previous-step estimate seeds the next step (deprecated)
        noise = torch.randn(n, d, generator=g, device=dev) * (dt**0.5)
        x = x + dt * drift + noise
        if return_path:
            path.append((float(ts[k + 1]), x.clone()))

    return (x, path) if return_path else x


def decode_drift_selfcond(x: torch.Tensor, net, h: torch.Tensor | None = None) -> torch.Tensor:
    """Round via the two-pass self_cond_predict at x, the "deployed" best estimate.

    A one-off extra forward pass (unlike sl_sample_selfcond's per-step call, which
    stays at one pass/step for cost reasons) -- worth it here since decode only runs
    once, at the very end.
    """
    d = x.shape[1]
    h = torch.zeros(d) if h is None else h
    hard = torch.sign(net.natural_self_cond(
        (h if h.dim() == 2 else h.unsqueeze(0)) + x))
    hard[hard == 0] = 1.0
    return hard


# ---------------------------------------------------------------------------
# Conditional sampling: given some coordinates, produce the rest.
# ---------------------------------------------------------------------------

@torch.no_grad()
def sl_sample_infill(net, d: int, known: torch.Tensor, values: torch.Tensor,
                     mode: str = "path", tau: float = 32.0,
                     T: float = 20.0, K: int = 200, schedule: str = "late",
                     seed: int = 0, device=None, self_cond_mode: str = "two_pass"):
    """Sample x ~ pi( . | x_known = values), two ways.

    known:  (n, d) bool  -- which coordinates are given
    values: (n, d) float -- their {-1,+1} values (ignored where known is False)

    mode="tilt"  -- THE NATURAL-PARAMETER ROUTE, in the existing framework.
        Set h = tau * values on the known coordinates and 0 elsewhere, then run
        the ordinary sampler. This is not an ad-hoc trick: normalize_natural
        rescales h to ||v|| = sqrt(d), which puts magnitude sqrt(d/k) on the k
        known coordinates and 0 on the rest -- exactly the profile
        noising.sample_infill_q draws for mode 1, so the query shape is one the
        operator was trained on whenever mode1_prob > 0.

        Its limit is tau. The drift is evaluated at h + X_t, and X_t grows to
        about +/-T per coordinate (T=20 here), so a clamp much below that stops
        biting by the end of the trajectory. Pushing tau up costs range: the
        level the network sees is beta_eff = ||h||/sqrt(d) = tau*sqrt(k/d), so
        at k = d/2 that is 0.707*tau, and training only covered beta <= 50. The
        clamp is therefore never hard -- it is a strong prior that decays as the
        sample accumulates its own evidence.

    mode="path" -- THE NOISING ROUTE, exact rather than asymptotic.
        Stochastic localization is X_t = t*x + B_t, so a coordinate whose value
        is KNOWN has a directly simulable path: dX_i = s_i dt + dB_i. That is
        the same Euler loop with the drift on those coordinates replaced by
        s_i, since E[x_i | x_i = s_i] = s_i. One line.

        The consequences are worth stating. There is no tau and nothing to
        tune. The known coordinates carry exactly the signal-to-noise a real
        sample would at every t, so the network is never queried outside the
        range it saw. And the conditioning does not decay -- X_t,known keeps
        growing in step with X_t,free instead of being outrun by it.

        This is exact GIVEN an exact operator. With a learned one the error is
        whatever the operator's error is on anisotropic states, which is
        precisely what mode1_prob > 0 during training was for: mid-trajectory
        the known coordinates are large while the free ones are small, and a
        model trained only on isotropic beta would rarely have seen that.

    The two are worth running against each other because they fail differently:
    "tilt" degrades gracefully into ignoring the context, while "path" cannot
    ignore it and instead exposes any operator error at anisotropic states.
    """
    if mode not in ("tilt", "path"):
        raise ValueError(f"mode must be 'tilt' or 'path', got {mode!r}")
    _check_self_cond_mode(self_cond_mode)
    dev = torch.device(device) if device is not None else torch.device("cpu")
    n = known.shape[0]
    known = known.to(dev)
    values = values.to(dev).float()
    g = torch.Generator(device=dev).manual_seed(seed)

    if mode == "tilt":
        h = torch.where(known, tau * values, torch.zeros((), device=dev))
        return sl_sample_selfcond(net, d, n, h=h, T=T, K=K, schedule=schedule,
                                  seed=seed, device=dev,
                                  self_cond_mode=self_cond_mode), h

    ts = time_grid(T, K, schedule, h=torch.zeros(d), d=d)
    x = torch.zeros(n, d, device=dev)
    x_pred = torch.zeros(n, d, device=dev)
    for k in range(K):
        dt = float(ts[k + 1] - ts[k])
        if self_cond_mode == "two_pass":
            x_pred = net.natural(x, x_pred=None)
        drift = net.natural(x, x_pred=x_pred)
        if self_cond_mode == "carry":
            x_pred = drift
        # The whole of mode="path": a known coordinate's posterior mean IS its
        # value, so its increment is s_i dt + dB_i and its path is t*s_i + B_i.
        drift = torch.where(known, values, drift)
        x = x + dt * drift + torch.randn(n, d, generator=g, device=dev) * (dt ** 0.5)
    return x, torch.zeros(d, device=dev)
