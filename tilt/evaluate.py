"""Evaluation.

Three levels, in increasing order of what they tell you and decreasing order of
how far they scale:

  operator_error   RMSE against the exact enumerated G.  Only available while d
                   is small; the gold standard.
  invariance_gap   G depends on (v, a) only through h = beta(a) v, so the same h
                   re-expressed with a different a must give the same answer.
                   Needs NO ground truth, works at any d, and empirically tracks
                   the true error closely.  This is what you carry to scale.
  kl_bound         Trapezoid estimate of the Girsanov bound on the sampler's
                   KL, integrating the squared operator error against dt.  A
                   bound, not a measurement -- end-to-end sample quality lives
                   in sampler.py and is not wired in here yet.

G is (a scaled) gradient of a convex potential (log Z), so its Jacobian is the
conditional covariance and must be symmetric PSD.  Two more ground-truth-free
diagnostics fall out of that:

  jacobian_symmetry      d_ui G_j == d_uj G_i everywhere.  Already above.
  jacobian_diagonal      d_i G_i == a/(1-a^2) * (1 - G_i^2), i.e. the tilt
                         strength times the Bernoulli variance.  Checked in
                         the cleared-denominator form to stay stable as
                         a -> 1.  Cheap: one random coordinate per point.
  residual_decorrelation RKHS diagnostic: residuals x - f(u) should be
                         kernel-uncorrelated across points, not just
                         zero-mean.  Weakest of the checks -- an L2-trained
                         net passes it "by construction up to noise" -- so
                         use it only as an end-of-training confirmation,
                         never as a training-loss term.

In-distribution vs. OOD (reward-tilted)
----------------------------------------
Every metric below takes `mode` ("in_dist" or "ood") and `tau`:

  mode="in_dist" (default)  x is drawn from the target's own law, exactly as
                            during training.
  mode="ood"                x is drawn from the reward-tilted law
                            q_tau(x) propto pi(x) * exp(tau * x_star), which
                            requires a RewardAugmented target (anything with
                            a `tilt_direction` method). This tilt is a linear
                            shift of the natural parameter, h_extra = tau *
                            e_star (see `targets.RewardAugmented.tilt_direction`
                            and `noising.build_query`'s `h_extra` argument), so
                            both the exact ground truth and the query fed to
                            `net` must fold h_extra in -- that's what the
                            `_ground_truth` / `_query` helpers below do. This
                            probes the operator at inputs whose natural
                            parameter has been shifted away from what
                            training (tau=0) typically produces.

This is target-agnostic: any EnumerableTarget with a `tilt_direction` method
(any base pi lifted by `RewardAugmented`, whatever pi and whatever reward r)
goes through this exact same "ood" path -- there is nothing sudoku- or
Ising-specific here.  See `sudoku.SudokuRewardAugmented` for the sudoku case;
it plugs in here with zero changes to this file.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from .noising import a_of_beta, beta_of_a, build_query, vp_noise

_MODES = ("in_dist", "ood")


def _resolve_h_extra(target, mode: str, tau: float):
    """mode="in_dist" -> None.  mode="ood" -> h_extra = tau * e_star."""
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if mode == "in_dist":
        return None
    if not hasattr(target, "tilt_direction"):
        raise AttributeError(
            f"{type(target).__name__} has no tilt_direction(); mode='ood' "
            "requires a RewardAugmented target"
        )
    return target.tilt_direction(tau)


def _typical_inputs(target, beta: float, n: int, seed: int = 0,
                     mode: str = "in_dist", tau: float = 0.0, sample_from=None):
    """Draw (x, u, a, h_extra) at signal level `beta`.

    mode="in_dist": x ~ target's own law.
    mode="ood": x ~ the reward-tilted law q_tau (see module docstring);
        h_extra is the natural-parameter shift this tilt induces, or None
        for mode="in_dist".

    sample_from, if given, overrides WHICH distribution x is actually drawn
    from (e.g. a conditioned/infilled target from
    sudoku_infill.condition_target) -- but h_extra is still resolved from the
    ORIGINAL `target` (via tilt_direction), and ground truth (_ground_truth)
    always uses the ORIGINAL `target` too. Once h exists, the exact
    conditional mean is m(h) = E_target[X | h] regardless of how h was
    produced -- sampling from a different law only changes WHICH h's get
    tested (probing a specific region of h-space), never which operator is
    being tested. See sudoku_infill.infill_eval_suite.
    """
    h_extra = _resolve_h_extra(target, mode, tau)
    src = sample_from if sample_from is not None else target
    g = torch.Generator().manual_seed(seed)
    a = a_of_beta(torch.full((n, 1), float(beta)))
    if h_extra is None:
        x = src.sample(n, generator=g)
    else:
        x = src.sample_tilted(h_extra, n, generator=g)
    return x, vp_noise(x, a, generator=g), a, h_extra


def _ground_truth(target, u, a, h_extra=None):
    """Exact E[X|U=u] under the (possibly reward-tilted) target.  See module docstring."""
    h = beta_of_a(a) * u
    if h_extra is not None:
        h = h + h_extra
    return target.natural_mean(h)


def _query(net, u, a, h_extra=None, d=None):
    """Renormalise (u, a[, h_extra]) via build_query, then evaluate net.

    Identical to calling net(u, a) directly when h_extra is None, since
    build_query is the identity in that case -- so this is a no-op refactor
    for mode="in_dist" and the only correct path for mode="ood".
    """
    v, a_star = build_query(u, a, h_extra=h_extra, d=d)
    return net(v, a_star)


@torch.no_grad()
def operator_error(net, target, betas, n: int = 2048, seed: int = 0,
                   scale: float = 1.0, mode: str = "in_dist", tau: float = 0.0,
                   sample_from=None):
    """RMSE and max error against the exact operator.  scale>1 probes off-shell.

    sample_from, if given, draws x from a different distribution than
    `target` itself (e.g. a conditioned/infilled target) while ground truth
    stays pinned to `target.natural_mean` -- see _typical_inputs.
    """
    rows = []
    for b in betas:
        _, u, a, h_extra = _typical_inputs(target, b, n, seed, mode, tau, sample_from=sample_from)
        u = u * scale
        gt = _ground_truth(target, u, a, h_extra)
        pred = _query(net, u, a, h_extra, d=target.d)
        err = (pred - gt).abs()
        row = {"beta": b, "a": float(a[0, 0]),
               "rmse": float(err.pow(2).mean().sqrt()),
               "max": float(err.max()),
               "sse": float(err.pow(2).sum(1).mean()),
               "bayes_floor": float((1 - gt**2).sum(1).mean())}
        if mode == "ood":
            row["tau"] = tau
        if mode == "in_dist" and hasattr(target, "mean_field_0"):
            row["mf0_rmse"] = float((target.mean_field_0(u, a) - gt).pow(2).mean().sqrt())
        rows.append(row)
    return rows


@torch.no_grad()
def operator_error_by_lambda(net, target, betas, lambdas, n: int = 2048, seed: int = 0):
    """Like operator_error, but at several fixed Lambda-focusing levels.

    For each lambda in `lambdas` (each must satisfy 1.0 <= lambda < sqrt(d),
    see noising.lambda_matrix), samples in-distribution x, rescales by the
    corresponding diagonal Lambda before VP-noising it -- exactly mirroring
    BatchSampler's per-batch lambda augmentation during training -- and
    compares the exact operator (evaluated at the Lambda-adjusted natural
    parameter beta(a) * Lambda @ u) against net's prediction (via
    build_query, which folds Lambda in the same way training does).
    lambda=1.0 recovers plain operator_error (Lambda=identity).

    Useful as a training diagnostic distinct from `mode="ood"`: this probes
    accuracy as a direct function of the Lambda level actually swept during
    training (BatchSampler draws lambda ~ Uniform(1, lambda_max) every
    batch), rather than at a specific reward-tilted natural parameter.
    """
    from .noising import lambda_matrix

    rows = []
    g = torch.Generator().manual_seed(seed)
    for lam in lambdas:
        diag = lambda_matrix(float(lam), target.d)
        for b in betas:
            x = target.sample(n, generator=g)
            a = a_of_beta(torch.full((n, 1), float(b)))
            xs = diag * x
            u = vp_noise(xs, a, generator=g)
            h = beta_of_a(a) * (diag * u)
            gt = target.natural_mean(h)
            diagb = diag.unsqueeze(0).expand(n, -1)
            v, a_star = build_query(u, a, diagb, d=target.d)
            pred = net(v, a_star)
            err = (pred - gt).abs()
            rows.append({
                "lambda": float(lam), "beta": b,
                "rmse": float(err.pow(2).mean().sqrt()),
                "sse": float(err.pow(2).sum(1).mean()),
            })
    return rows


def sse_eval_fn(target, betas, n: int = 2048, seed: int = 0,
                 jac_random_n: int = 0, jac_diag_n: int = 0,
                 mode: str = "in_dist", tau: float = 0.0):
    """Build an eval_fn(net) -> dict for train()'s eval_fn hook / wandb logging.

    Reports SSE against the exact operator G (E||G - G_theta||^2, the "what
    actually matters" term of the loss decomposition -- see train.py's
    docstring) at each beta, plus the beta-averaged SSE as a single scalar to
    track overall progress on one curve. This is the test error against the
    true score, as opposed to the training loss which also carries the
    (large, constant) Bayes-floor term.

    If `jac_random_n > 0`, also runs `jacobian_symmetry_random` at each beta
    (ground-truth-free, so it's cheap insurance even once d is too large for
    the exact SSE above) and folds it into the same dict under
    "jac_symmetry_random", logged to wandb as
    test/jac_symmetry_random/<beta>/{mean_sq_diff,max_sq_diff,mean_rel_sq_diff}.

    If `jac_diag_n > 0`, also runs `jacobian_diagonal` at each beta
    (ground-truth-free) and folds it into the same dict under "jac_diagonal",
    logged to wandb as test/jac_diagonal/<beta>/{mean_violation,max_violation}.

    `mode`/`tau` select in-distribution (default) vs. reward-tilted OOD
    evaluation; see the module docstring.
    """
    def _fn(net):
        rows = operator_error(net, target, betas, n=n, seed=seed, mode=mode, tau=tau)
        out = {f"beta_{r['beta']:g}": {"sse": r["sse"], "rmse": r["rmse"]} for r in rows}
        out["mean_sse"] = sum(r["sse"] for r in rows) / len(rows)
        if jac_random_n > 0:
            out["jac_symmetry_random"] = jacobian_symmetry_random(
                net, target, betas, n=jac_random_n, seed=seed, mode=mode, tau=tau)
        if jac_diag_n > 0:
            out["jac_diagonal"] = jacobian_diagonal(
                net, target, betas, n=jac_diag_n, seed=seed, mode=mode, tau=tau)
        return out
    return _fn


@torch.no_grad()
def invariance_gap(net, target, betas, ratio: float = 2.0, n: int = 2048, seed: int = 0,
                    mode: str = "in_dist", tau: float = 0.0):
    """Self-consistency of the operator under reparameterisation. No ground truth.

    Re-express h = beta(a1) u1 as (u2, a2) with beta(a2) = ratio * beta(a1);
    a perfect operator returns identical values. h_extra (mode="ood") is a
    constant shared by both representations of h, so the u2 reparameterisation
    formula is unchanged -- only the query fed to `net` needs it folded in.
    """
    rows = []
    for b in betas:
        _, u1, a1, h_extra = _typical_inputs(target, b, n, seed, mode, tau)
        a2 = a_of_beta(torch.full_like(a1, float(b * ratio)))
        u2 = (beta_of_a(a1) / beta_of_a(a2)) * u1
        gap = (_query(net, u1, a1, h_extra, d=target.d)
               - _query(net, u2, a2, h_extra, d=target.d)).pow(2).mean().sqrt()
        row = {"beta": b, "gap": float(gap)}
        if mode == "ood":
            row["tau"] = tau
        rows.append(row)
    return rows


def jacobian_symmetry(net, target, beta: float, n: int = 16, seed: int = 0,
                       mode: str = "in_dist", tau: float = 0.0):
    """G(.,a) = grad of a convex potential, so dG/du must be symmetric PSD.

    Reports relative asymmetry and the most negative eigenvalue. No ground truth.
    """
    _, u, a, h_extra = _typical_inputs(target, beta, n, seed, mode, tau)
    asym, min_eig = [], []
    for i in range(n):
        ui = u[i : i + 1].clone().requires_grad_(True)
        J = torch.autograd.functional.jacobian(
            lambda z: _query(net, z, a[i : i + 1], h_extra, d=target.d).squeeze(0),
            ui, vectorize=True).squeeze(1)
        asym.append(float((J - J.T).norm() / (J.norm() + 1e-12)))
        min_eig.append(float(torch.linalg.eigvalsh(0.5 * (J + J.T)).min()))
    result = {"beta": beta, "rel_asymmetry": sum(asym) / n, "min_eig": min(min_eig)}
    if mode == "ood":
        result["tau"] = tau
    return result


def jacobian_symmetry_random(net, target, betas, n: int = 256, seed: int = 0,
                              mode: str = "in_dist", tau: float = 0.0):
    """Randomized off-diagonal check: d_j G_i == d_i G_j at one random (i, j) pair per point.

    Cheaper alternative to `jacobian_symmetry`'s full d x d Jacobian: only two
    scalar backward passes per point (one for d_j G_i, one for d_i G_j), so n
    can be pushed much higher for the same cost, trading a dense per-point
    check for a broad, randomized sweep across coordinates/locations/betas.
    Reports the mean and max squared difference (d_j G_i - d_i G_j)^2, plus
    the same normalized by ||G||^2 so it's comparable across betas/scales.
    """
    rows = []
    for b in betas:
        _, u, a, h_extra = _typical_inputs(target, b, n, seed, mode, tau)
        idx_gen = torch.Generator().manual_seed(seed + 1)
        # sample i != j for each point
        i_idx = torch.randint(0, target.d, (n,), generator=idx_gen)
        offset = torch.randint(1, target.d, (n,), generator=idx_gen)
        j_idx = (i_idx + offset) % target.d

        sq_diff, sq_norm = [], []
        for k in range(n):
            i, j = int(i_idx[k]), int(j_idx[k])
            with torch.enable_grad():
                uk = u[k : k + 1].clone().requires_grad_(True)
                ak = a[k : k + 1]
                gk = _query(net, uk, ak, h_extra, d=target.d)
                (grad_i,) = torch.autograd.grad(gk[0, i], uk, create_graph=False, retain_graph=True)
                (grad_j,) = torch.autograd.grad(gk[0, j], uk)
            with torch.no_grad():
                d_j_gi = grad_i[0, j]
                d_i_gj = grad_j[0, i]
                sq_diff.append(float((d_j_gi - d_i_gj) ** 2))
                sq_norm.append(float(gk.detach().pow(2).sum()) + 1e-12)
        row = {
            "beta": b,
            "mean_sq_diff": sum(sq_diff) / n,
            "max_sq_diff": max(sq_diff),
            "mean_rel_sq_diff": sum(d / s for d, s in zip(sq_diff, sq_norm)) / n,
        }
        if mode == "ood":
            row["tau"] = tau
        rows.append(row)
    return rows


def jacobian_diagonal(net, target, betas, n: int = 16, seed: int = 0,
                       mode: str = "in_dist", tau: float = 0.0):
    """Diagonal self-consistency: d_i G_i(u,a) == a/(1-a^2) * (1 - G_i(u,a)^2).

    G_i is the mean of a {-1,1} variable and 1 - G_i^2 its variance, so the
    diagonal Jacobian entry must equal the tilt strength times that variance.
    Uses the cleared-denominator form to avoid the inf*0 blowup as a -> 1:

        ((1 - a^2) * d_i g_i  -  a * (1 - g_i^2))^2

    One randomly chosen coordinate i per probed point (cheaper than the full
    Jacobian in jacobian_symmetry).  No ground truth needed.
    """
    rows = []
    for b in betas:
        _, u, a, h_extra = _typical_inputs(target, b, n, seed, mode, tau)
        idx_gen = torch.Generator().manual_seed(seed + 1)
        idx = torch.randint(0, target.d, (n,), generator=idx_gen)
        viol = []
        for k in range(n):
            i = int(idx[k])
            with torch.enable_grad():
                uk = u[k : k + 1].clone().requires_grad_(True)
                ak = a[k : k + 1]
                gk = _query(net, uk, ak, h_extra, d=target.d)
                gi = gk[0, i]
                (grad_i,) = torch.autograd.grad(gi, uk)
            with torch.no_grad():
                d_i = grad_i[0, i]
                t = ak[0, 0]
                viol.append(float(((1 - t**2) * d_i - t * (1 - gi**2)) ** 2))
        row = {"beta": b, "mean_violation": sum(viol) / n, "max_violation": max(viol)}
        if mode == "ood":
            row["tau"] = tau
        rows.append(row)
    return rows


@torch.no_grad()
def residual_decorrelation(net, target, betas, n: int = 256, bandwidth: float = 1.0,
                            seed: int = 0, mode: str = "in_dist", tau: float = 0.0):
    """RKHS diagnostic: residuals should be kernel-uncorrelated, not just mean-zero.

    Statistic: (1/(n(n-1))) * sum_{i != j} <r_i, r_j> * kappa(u_i, u_j), with
    r_i = x_i - G_theta(u_i, a_i) and a Gaussian kernel
    kappa(u, u') = exp(-0.5 ||u - u'||^2 / bandwidth^2). Zero at the true
    optimum, but weak -- an L2-trained net passes this "by construction up to
    noise," so treat it as an end-of-training confirmation only, never as a
    training-loss term.
    """
    rows = []
    for b in betas:
        x, u, a, h_extra = _typical_inputs(target, b, n, seed, mode, tau)
        r = x - _query(net, u, a, h_extra, d=target.d)
        d2 = torch.cdist(u, u).pow(2)
        kappa = torch.exp(-0.5 * d2 / bandwidth**2)
        gram = r @ r.T
        off_mask = ~torch.eye(n, dtype=torch.bool)
        stat = (gram * kappa)[off_mask].mean()
        row = {"beta": b, "residual_decorrelation": float(stat)}
        if mode == "ood":
            row["tau"] = tau
        rows.append(row)
    return rows


def plot_lollipop(true_G, net_G, ax=None, title: str | None = None):
    """Two-color stem plot: true G vs. net G, per coordinate.

    A quick at-a-glance diagnostic for *where* a network's predictions
    diverge from ground truth (as opposed to the aggregate RMSE/SSE numbers
    from `operator_error`). Accepts either a single point -- `true_G`/`net_G`
    of shape (d,), drawn on one panel -- or a batch of shape (n, d), drawn as
    n stacked panels sharing the coordinate axis (one row per sample point,
    labeled "pt 0", "pt 1", ...).

    Blue circles/stems are `true_G`, red x's/stems are `net_G`; both are
    clamped to no particular range but G itself always lies in [-1, 1].
    Returns (fig, axes) for further tweaking/saving.
    """
    import matplotlib.pyplot as plt

    true_G = torch.as_tensor(true_G).detach().cpu()
    net_G = torch.as_tensor(net_G).detach().cpu()
    if true_G.shape != net_G.shape:
        raise ValueError(
            f"shape mismatch: true_G {tuple(true_G.shape)} vs net_G {tuple(net_G.shape)}"
        )
    if true_G.ndim == 1:
        true_G = true_G.unsqueeze(0)
        net_G = net_G.unsqueeze(0)
    if true_G.ndim != 2:
        raise ValueError(f"expected true_G/net_G of shape (d,) or (n, d), got ndim={true_G.ndim}")

    n, d = true_G.shape
    coords = torch.arange(d)

    if ax is not None:
        if n != 1:
            raise ValueError("a single `ax` can only hold one point; pass ax=None to plot a batch")
        fig = ax.figure
        axes = [ax]
    else:
        fig, axes = plt.subplots(n, 1, figsize=(10, 1.6 * n), sharex=True, squeeze=False)
        axes = list(axes[:, 0])

    for row, cur_ax in enumerate(axes):
        tg, ng = true_G[row], net_G[row]
        cur_ax.vlines(coords, 0, tg, color="tab:blue", lw=1.5, zorder=1)
        cur_ax.vlines(coords, 0, ng, color="tab:red", lw=1.0, zorder=2)
        cur_ax.plot(coords, tg, "o", color="tab:blue", ms=5,
                    label="true G" if row == 0 else None, zorder=3)
        cur_ax.plot(coords, ng, "x", color="tab:red", ms=6, mew=1.5,
                    label="net G" if row == 0 else None, zorder=4)
        cur_ax.axhline(0, color="black", lw=0.5)
        cur_ax.set_ylim(-1.1, 1.1)
        if n > 1:
            cur_ax.set_ylabel(f"pt {row}", rotation=0, ha="right", va="center")

    axes[-1].set_xlabel("coordinate")
    axes[0].legend(loc="upper right")
    if title:
        if n == 1:
            axes[0].set_title(title)
        else:
            fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def kl_bound(net, target, betas, n: int = 2048, seed: int = 0,
             mode: str = "in_dist", tau: float = 0.0):
    """Trapezoid estimate of the Girsanov bound 0.5 * int E||b - b_hat||^2 dt.

    Uses SL time t = beta * a(beta) as the integration variable.
    """
    from .noising import sl_time_of_beta

    rows = operator_error(net, target, betas, n=n, seed=seed, mode=mode, tau=tau)
    ts = [float(sl_time_of_beta(torch.tensor(float(b)))) for b in betas]
    sq = [r["rmse"] ** 2 * target.d for r in rows]
    total = sum(0.5 * (sq[i] + sq[i + 1]) / 2 * (ts[i + 1] - ts[i])
                for i in range(len(ts) - 1))
    return {"kl_bound_nats": total, "tv_pinsker": math.sqrt(max(total, 0.0) / 2),
            "t_range": (ts[0], ts[-1])}
