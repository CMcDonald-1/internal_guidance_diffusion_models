r"""Training loop for SelfCondTiltMLP -- a standalone parallel to train.py.

Identical to train.py's train() (same BatchSampler, build_query, optimizer/schedule,
wandb logging, checkpoint format) except for the self-conditioning coin flip each step,
following Bit Diffusion's Algorithm 1: with probability self_cond_prob, compute
x_pred = f(u, 0, a) (stop-gradient) and train on f(u, x_pred, a); otherwise train on
f(u, 0, a) directly, so the network also learns to work well with no prior estimate
(needed for the very first sampling step). This file never imports/modifies train.py;
it duplicates the small amount of loop structure needed so non-self-cond training is
completely unaffected by anything here.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from . import checkpointing as _ckpt
from .data import BatchSampler, holdout_loss as _holdout_loss, load_dataset
from .models import MLPConfig
from .models_selfcond import SelfCondTiltMLP
from .noising import build_query, dt_dlogbeta
from .train import _flatten_metrics, _sectioned_test_log


def train_selfcond(datadir: str | Path, outpath: str | Path, cfg: MLPConfig | None = None,
                    steps: int = 3000, batch_size: int = 1024, lr: float = 1e-3,
                    self_cond_prob: float = 0.5,
                    reward_weight: float = 1.0,
                    dt_weight: bool = False, lambda_max: float = 1.0,
                    mode1_prob: float = 0.0, lambda1_prob: float = 0.25,
                    beta_focus_range: tuple[float, float] | None = None,
                    beta_focus_prob: float = 0.0, seed: int = 0,
                    log_every: int = 500, verbose: bool = True,
                    eval_fn=None, eval_every: int | None = None,
                    holdout: dict | None = None, holdout_every: int | None = None,
                    use_wandb: bool = False, wandb_project: str = "tilted-diffusions",
                    wandb_run_name: str | None = None, wandb_config: dict | None = None,
                    wandb_entity: str | None = None, wandb_tags: list[str] | None = None,
                    wandb_notes: str | None = None,
                    device: str | torch.device | None = None,
                    ckpt_every: int | None = None, resume: bool = False,
          net=None):
    """See train.py::train()'s docstring for eval_fn/holdout/device/wandb semantics --
    all unchanged here, except holdout loss is scored via net.self_cond_predict (the
    deployed two-pass estimate), not a plain 2-arg forward call -- same reasoning as
    eval_fn, which is expected to come from evaluate_selfcond.py for the same reason
    (a plain forward call would silently default x_pred to zero and understate the
    model's real accuracy).

    self_cond_prob is the per-step (not per-example) probability of running the
    two-pass self-conditioning update; 0.0 degenerates to training an ordinary
    (x_pred=0 always) SelfCondTiltMLP, 1.0 always self-conditions (never trains the
    x_pred=0 fallback path).

    reward_weight multiplies the squared error on the LAST coordinate (x_*) only.
    1.0 is exactly the old behaviour -- the multiply is skipped entirely, so every
    pre-existing run is bit-identical.

    WHY THIS KNOB EXISTS
    --------------------
    The loss is ||x - x_hat||^2 SUMMED over d coordinates, so x_* is 1 term out of
    481. Price what solving it is worth: a +/-1 coordinate at density p costs
    4p(1-p) if you predict only the marginal and ignore the tokens entirely --
    0.1536 at the loose arm's 4% injection, 0.0278 at the strict arm's 0.7%.
    Against an observed train loss of ~228 that is 0.067% of the objective, and
    it is 0.32x what an AVERAGE single coordinate costs (228/481 = 0.474). The
    reward coordinate is the cheapest coordinate in the vector to give up on, and
    it is the one that requires a CMUdict rhyme relation across positions 16
    apart. The optimiser is behaving correctly; the objective is asking for the
    wrong thing.

    That accounts for six failed levers at once. tau and K are sampling knobs and
    cannot change a training incentive. Steps (50k->100k) and capacity (0.57x
    ->2.66x params) give more of both and leave the incentive alone. Density
    (0.7%->5%) moves the prize from 0.028 to 0.19, still noise against 228, which
    is exactly why it did nothing. Per-token flags coupled at 99.9% because a
    same-slot table lookup is nearly free -- not because the objective rewarded
    them.

    Rough calibration at the loose arm's 4%: reward_weight=45 puts x_* at ~3% of
    the loss, 150 at ~9%, 500 at ~25%.

    WHAT TO WATCH, and it does not need a sample eval
    ------------------------------------------------
    train/reward_mse is the UNWEIGHTED squared error on x_* alone, and
    train/reward_mse_ratio divides it by that 4p(1-p) marginal-only floor. Ratio
    ~1.0 means the model is still predicting the base rate and ignoring the
    tokens -- the failure, visible within a few thousand steps. Below ~0.8 means
    x_* has become a function of the other coordinates, which is the entire
    hypothesis. Do not wait for a sampling run to find this out.

    The holdout loss is deliberately NOT reweighted, so holdout_mean_sse stays
    comparable against every run that predates this flag.
    
    net, if given, is used INSTEAD of constructing one from cfg. This is the hook
    that lets an alternative featuriser (e.g. tilt.models_transformer's
    TransformerTilt) reuse this loop unchanged: the loop only ever calls the net,
    never inspects its class. cfg must still be passed and its `d` must match the
    dataset. Leave net=None for the default behaviour.
    """
    torch.manual_seed(seed)
    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    x, frozen, target, meta = load_dataset(datadir)
    cfg = cfg or MLPConfig(d=meta["d"])
    if cfg.d != meta["d"]:
        raise ValueError(f"config d={cfg.d} != dataset d={meta['d']}")

    # Coordinates past the token blocks: 1 for every reward-bearing run, 0 for an
    # untilted run (owt_v1). MLPConfig has no n_extra, hence the getattr default,
    # which reproduces every pre-existing call exactly.
    n_extra = int(getattr(cfg, "n_extra", 1))
    batches = BatchSampler(x, meta["beta_lo"], meta["beta_hi"], frozen,
                           lambda_max=lambda_max, mode1_prob=mode1_prob,
                           lambda1_prob=lambda1_prob, beta_focus_range=beta_focus_range,
                           beta_focus_prob=beta_focus_prob, seed=seed,
                           n_extra=n_extra)
    net = (net if net is not None else SelfCondTiltMLP(cfg)).to(device)

    # Per-coordinate loss weights. A vector, not an in-place write on the squared
    # error: pow(2)'s backward needs its INPUT, so mutating its output happens to
    # be safe, but it is the kind of safe that stops being safe when someone adds
    # a second consumer of that tensor.
    coord_w = None
    if reward_weight != 1.0:
        if n_extra == 0:
            raise ValueError(
                f"reward_weight={reward_weight} with n_extra=0: there is no x_* "
                f"coordinate to weight. This would silently up-weight the last "
                f"token bit instead.")
        coord_w = torch.ones(cfg.d, device=device)
        coord_w[-1] = reward_weight
    # The marginal-only floor for x_*: what the model pays by predicting the base
    # rate and ignoring every token. success_rate is the INJECTED training density
    # (0.04 for the loose arm), which is the right p here -- the natural pool rate
    # is not what the model sees.
    p_rw = meta.get("success_rate") if n_extra else None
    rw_floor = 4.0 * p_rw * (1.0 - p_rw) if p_rw else None
    if verbose:
        print(f"  n_extra {n_extra}  reward_weight {reward_weight}"
              + (f"   x_* marginal-only floor 4p(1-p) = {rw_floor:.4f} at p={p_rw:.4f}"
                 if rw_floor else "   (no success_rate in meta; no floor to compare)"),
              flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    eval_every = eval_every or log_every
    holdout_every = holdout_every or eval_every
    if holdout is not None:
        holdout = {k: (v.to(device) if v is not None else None) for k, v in holdout.items()}
    history, test_history, holdout_history, t0 = [], [], [], time.time()
    coin_gen = torch.Generator().manual_seed(seed + 1)  # CPU generator; used for one scalar coin flip per step

    # See train.py for the contract. coin_gen must travel too: it drives the
    # per-step self-conditioning coin flip, so a resume that reseeds it takes a
    # different sequence of one-pass/two-pass updates from step start_step on.
    start_step = 1
    if resume:
        start_step, restored = _ckpt.load_partial(
            outpath, net, opt, sched, batches,
            extra_gens={"coin_gen": coin_gen}, verbose=verbose)
        if restored:
            history = restored.get("history", history)
            test_history = restored.get("test_history", test_history)
            holdout_history = restored.get("holdout_history", holdout_history)

    wandb_run = None
    if use_wandb:
        import wandb
        run_config = {
            "d": cfg.d, "width": getattr(cfg, "width", None),
            "n_blocks": getattr(cfg, "n_blocks", None),
            "skip": cfg.skip, "arch": type(net).__name__,
            "cfg_tag": getattr(cfg, "tag", None),
            "steps": steps, "batch_size": batch_size, "lr": lr, "dt_weight": dt_weight,
            "self_cond_prob": self_cond_prob, "reward_weight": reward_weight,
            "reward_mse_floor": rw_floor,
            "lambda_max": lambda_max, "mode1_prob": mode1_prob,
            "lambda1_prob": lambda1_prob,
            "beta_focus_range": beta_focus_range, "beta_focus_prob": beta_focus_prob,
            "seed": seed,
            "beta_lo": meta["beta_lo"], "beta_hi": meta["beta_hi"],
            "dataset": Path(datadir).name,
        }
        if wandb_config:
            run_config.update(wandb_config)
        wandb_run = wandb.init(project=wandb_project, entity=wandb_entity,
                                name=wandb_run_name, config=run_config,
                                tags=wandb_tags, notes=wandb_notes)

    for step in range(start_step, steps + 1):
        xb, ub, ab, bb, diagb = batches(batch_size)
        xb, ub, ab, bb = xb.to(device), ub.to(device), ab.to(device), bb.to(device)
        if diagb is not None:
            diagb = diagb.to(device)
        vb, ab_star = build_query(ub, ab, diagb, d=cfg.d)

        if float(torch.rand((), generator=coin_gen)) < self_cond_prob:
            with torch.no_grad():
                x0 = net(vb, ab_star, x_pred=None)
            pred = net(vb, ab_star, x_pred=x0)
        else:
            pred = net(vb, ab_star, x_pred=None)

        sq = (xb - pred) ** 2
        per = (sq * coord_w).sum(1) if coord_w is not None else sq.sum(1)
        if dt_weight:
            w = dt_dlogbeta(bb).squeeze(1)
            loss = (w * per).sum() / w.sum()
        else:
            loss = per.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            # UNWEIGHTED error on x_* alone, and its ratio to the marginal-only
            # floor. This is the readout for the whole reward-weight hypothesis
            # and it is available every log_every steps instead of after a
            # 10-hour run plus a sampling job. Ratio ~1.0 = still predicting the
            # base rate; well below 1 = x_* became a function of the tokens.
            # .detach(): float() on a tensor that requires grad warns, and this
            # line runs every log_every steps for the whole run.
            # With n_extra=0 the last coordinate is an ordinary token bit, not
            # x_*, so reporting it as reward_mse would be actively misleading.
            rmse = float(sq[:, -1].detach().mean()) if n_extra else float("nan")
            ratio = (rmse / rw_floor) if rw_floor else float("nan")
            history.append({"step": step, "loss": float(loss),
                            "reward_mse": rmse, "reward_mse_ratio": ratio})
            if verbose:
                print(f"  step {step:>6}  loss {float(loss):.4f}"
                      f"  x*_mse {rmse:.4f} ({ratio:.2f}x floor)"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
            if wandb_run is not None:
                wandb_run.log({"train/loss": float(loss),
                               "train/reward_mse": rmse,
                               "train/reward_mse_ratio": ratio,
                               "lr": sched.get_last_lr()[0]}, step=step)
        if holdout is not None and (step % holdout_every == 0 or step == steps):
            net.eval()
            hl = _holdout_loss(net, holdout, d=cfg.d, self_cond=True)
            net.train()
            holdout_history.append({"step": step, "holdout_mean_sse": hl})
            if verbose:
                print(f"    holdout {step:>6}  mean_sse {hl:.4f}", flush=True)
            if wandb_run is not None:
                wandb_run.log({"test-holdout/mean_sse": hl}, step=step)
        if eval_fn is not None and (step % eval_every == 0 or step == steps):
            net.eval()
            if device.type != "cpu":
                net.to("cpu")
            with torch.no_grad():
                metrics = eval_fn(net)
            if device.type != "cpu":
                net.to(device)
            net.train()
            test_history.append({"step": step, **metrics})
            if verbose:
                if "mean_sse" in metrics:
                    print(f"    eval {step:>6}  mean_sse {metrics['mean_sse']:.4f}", flush=True)
                else:
                    print(f"    eval {step:>6}  {metrics}", flush=True)
            if wandb_run is not None:
                flat = _flatten_metrics(metrics)
                wandb_run.log(_sectioned_test_log(flat), step=step)

        if ckpt_every and step % ckpt_every == 0 and step != steps:
            _ckpt.save_partial(outpath, net, opt, sched, batches, step,
                               {"history": history, "test_history": test_history,
                                "holdout_history": holdout_history},
                               extra_gens={"coin_gen": coin_gen},
                               meta={"datadir": str(datadir), "steps": steps,
                                     "seed": seed})

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    net.save(outpath, extra={
        "datadir": str(datadir), "steps": steps, "batch_size": batch_size,
        "lr": lr, "dt_weight": dt_weight, "self_cond_prob": self_cond_prob,
        "reward_weight": reward_weight, "reward_mse_floor": rw_floor,
        "lambda_max": lambda_max,
        "mode1_prob": mode1_prob, "lambda1_prob": lambda1_prob,
        "beta_focus_range": beta_focus_range, "beta_focus_prob": beta_focus_prob,
        "seed": seed, "history": history, "test_history": test_history,
        "holdout_history": holdout_history,
        "beta_lo": meta["beta_lo"], "beta_hi": meta["beta_hi"],
    })
    net.test_history = test_history
    net.holdout_history = holdout_history
    _ckpt.clear_partial(outpath, verbose=verbose)
    if verbose:
        print(f"  saved -> {outpath}")
    if wandb_run is not None:
        wandb_run.summary["final_train_loss"] = history[-1]["loss"] if history else None
        wandb_run.summary["train_time_s"] = time.time() - t0
        wandb_run.finish()
    return net
