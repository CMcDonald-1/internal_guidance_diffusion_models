"""Reparameterizations between the three coordinate systems for the tilt.

The tilt mean operator depends on its arguments ONLY through the natural
parameter h:

    G(v, a) = m(h),    h = beta(a) * v,    beta(a) = a / (1 - a^2)

Three equivalent ways to name the same tilt:

  natural   h in R^d          unbounded, the fundamental one
  VP        (v, a)            v of arbitrary norm, a in [0,1)
  normal'd  (sqrt(d)*h_hat, t*)   fixed-norm direction + scalar time

The last is the one the write-up derives via the quadratic
    t/(1-t^2) = C,  C = ||h|| / sqrt(d)   =>   t* = (sqrt(1/C^2 + 4) - 1/C)/2
and it is what both training and sampling should feed the network, so the
invariance G(v,a) = G(cv,a') is respected by construction rather than learned.
"""
from __future__ import annotations

import math

import torch

EPS = 1e-12


def beta_of_a(a: torch.Tensor) -> torch.Tensor:
    """Tilt strength beta = a / (1 - a^2)."""
    return a / (1.0 - a**2).clamp_min(EPS)


def a_of_beta(b: torch.Tensor) -> torch.Tensor:
    """Inverse of beta_of_a: the positive root of a^2 + a/beta - 1 = 0."""
    return (-1.0 + torch.sqrt(1.0 + 4.0 * b**2)) / (2.0 * b.clamp_min(EPS))


def sl_time_of_beta(b: torch.Tensor) -> torch.Tensor:
    """Stochastic-localization time t corresponding to tilt strength beta.

    Under X_t = t X + sqrt(t) Z the natural parameter is h = X_t, whose scale
    grows like t; matching to the VP family gives t = beta * a(beta).
    """
    return b * a_of_beta(b)


def build_query(u: torch.Tensor, a: torch.Tensor, diag: torch.Tensor | None = None,
                h_extra: torch.Tensor | None = None, d: int | None = None):
    """Turn a raw noisy observation into the correctly renormalised (v*, a*) query for G.

    u is the RAW noisy observation U = a * Lambda * X + sqrt(1-a^2) Z (Lambda = I if diag is
    None).  Because ||Lambda x||^2 = d is constant on the cube, the natural parameter implied
    by observing u is beta(a) * Lambda @ u -- NOT beta(a) * u -- so Lambda must be folded in
    before the network is queried.  h_extra adds any further additive natural parameter (e.g.
    tau * e_star for a reward tilt) before the same renormalisation is applied, so every kind
    of tilt -- noise-induced, Lambda-augmented, reward -- goes through one consistent path.
    """
    if diag is None and h_extra is None:
        return u, a  # nothing to fold in -- raw (u, a) IS already the correct query
    h = beta_of_a(a) * u if diag is None else beta_of_a(a) * (diag * u)
    if h_extra is not None:
        h = h + h_extra
    return normalize_natural(h, d=d)


def normalize_natural(h: torch.Tensor, d: int | None = None):
    """Natural parameter -> (fixed-norm direction, time) network inputs.

    Returns (v, a) with ||v|| = sqrt(d) and beta(a) * v == h.
    """
    d = h.shape[-1] if d is None else d
    norm = h.norm(dim=-1, keepdim=True).clamp_min(EPS)
    C = norm / math.sqrt(d)
    a = a_of_beta(C)
    v = math.sqrt(d) * h / norm
    return v, a


def vp_noise(x: torch.Tensor, a: torch.Tensor, generator=None) -> torch.Tensor:
    """Variance-preserving noising U = a X + sqrt(1 - a^2) Z."""
    z = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return a * x + torch.sqrt((1.0 - a**2).clamp_min(0.0)) * z


def sample_beta_loguniform(n: int, lo: float, hi: float, generator=None) -> torch.Tensor:
    """Log-uniform tilt strengths, shape (n, 1)."""
    u = torch.rand(n, 1, generator=generator)
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * u)


def dt_dlogbeta(b: torch.Tensor) -> torch.Tensor:
    """d t / d log(beta), the Girsanov reweighting factor.

    The KL bound integrates the squared drift error against dt, but training
    samples log-uniformly in beta.  Multiplying the per-sample loss by this
    factor makes the empirical objective an unbiased estimate of the bound.
    """
    b = b.clamp_min(EPS)
    a = a_of_beta(b)
    r = torch.sqrt(1.0 + 4.0 * b**2)
    da_dlogb = b * (2.0 * b / r - (r - 1.0) / (2.0 * b**2)) / 1.0
    # t = beta * a  =>  dt/dlogbeta = beta * (a + beta * da/dbeta)
    da_db = (2.0 * b / r * (2.0 * b) - (r - 1.0)) / (2.0 * b**2)
    return b * (a + b * da_db)


def lambda_matrix(lam: float, d: int, device=None) -> torch.Tensor:
    """Diagonal Lambda from the write-up: emphasises the last coordinate.

    Lambda_jj = sqrt(1 + 1/(d-1) - lam^2/(d-1)) for j < d, Lambda_dd = lam,
    chosen so ||Lambda x||^2 = d on the cube.  lam in [1, sqrt(d)];
    lam = 1 gives the identity, lam = sqrt(d) is singular (excluded).

    Special case of profile_diag(lam, q=ones(d-1), d) below -- kept as its own
    function since operator_error_by_lambda (evaluate.py) only ever needs the
    plain reward-focusing profile and a scalar lam.
    """
    if not 1.0 <= lam < math.sqrt(d):
        raise ValueError(f"lam must lie in [1, sqrt(d)) = [1, {math.sqrt(d):.3f})")
    head = math.sqrt(1.0 + (1.0 - lam**2) / (d - 1))
    diag = torch.full((d,), head, device=device)
    diag[-1] = lam
    return diag


def profile_diag(lam: torch.Tensor, q: torch.Tensor, n_extra: int = 1) -> torch.Tensor:
    """General batched Lambda diagonal from a core profile q and reward-block scale lam.

    q: (..., core) nonnegative "core" profile (everything but the reward block)
    with sum(q**2, dim=-1) == core -- e.g. all-ones for plain reward-focusing
    (recovers lambda_matrix above at n_extra=1), or k nonzero entries of
    sqrt(core/k) each and the rest 0 for infill-focusing on a random k-subset
    of coordinates (write-up Section 8.3's "mode 1").  lam: (..., 1) reward
    stretch, applied to ALL n_extra reward coordinates as one block.

    n_extra widens the reward block (mnist_v1: 10 one-hot class bits). The
    budget identity ||ell||^2 == d = core + n_extra becomes

        core * scale + n_extra * lam^2 == d,
        scale = 1 + n_extra * (1 - lam^2) / core,

    so the admissible range is lam in [1, sqrt(d / n_extra)) -- NOT sqrt(d).
    At n_extra=1 this is the historical formula, bit-for-bit: same scale, and
    expand() of a (..., 1) lam is a no-op. Note the t=0 sampler query (all
    norm on the reward block, i.e. the pattern tilt tau * r_c) sits exactly at
    the singular endpoint sqrt(d / n_extra), so the sudoku6 coverage lesson
    (train close beneath the endpoint) transfers with the NEW ceiling.

    Returns the full (..., d) diagonal ell with ell[..., :core] =
    sqrt(scale) * q and ell[..., core:] = lam on every reward coordinate, so
    ||ell||^2 == d on the cube for *any* admissible q and lam -- this is what
    lets q and lam vary independently, and per-row, in a batch.
    """
    core = q.shape[-1]
    scale = (1.0 + n_extra * (1.0 - lam**2) / core).clamp_min(0.0)
    core_diag = torch.sqrt(scale) * q
    return torch.cat([core_diag, lam.expand(*lam.shape[:-1], n_extra)], dim=-1)


def sample_infill_q(bs: int, core: int, generator=None) -> torch.Tensor:
    """Batch of sparse infill-mode core profiles (write-up Section 8.4 steps 2-3).

    Per row: draw k uniform on {2, ..., core}, pick a uniformly random k-subset
    of the core coordinates, set those entries to sqrt(core/k) and the rest to
    0 -- so sum(q**2) == core for every row regardless of k, satisfying
    profile_diag's budget requirement. Vectorised via per-row random ranks
    rather than a Python loop.
    """
    scores = torch.rand(bs, core, generator=generator)
    ranks = scores.argsort(dim=1).argsort(dim=1)
    k = torch.randint(2, core + 1, (bs, 1), generator=generator).float()
    mask = (ranks < k).float()
    return mask * torch.sqrt(core / k)
