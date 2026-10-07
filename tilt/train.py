r"""Training: loads a dataset + model config, minimises the denoising loss.

    L = E || X - G_theta(U, a) ||^2
      = E || X - G ||^2   +   E || G - G_theta ||^2
        \_______________/     \___________________/
         Bayes floor, const     what actually matters

The floor is large and constant, so the training loss is a terrible progress
signal -- the informative quantity is the second term, which evaluate.py
measures directly against the exact operator.

--dt-weight reweights each sample by dt/dlog(beta).  The Girsanov bound on the
sampler's KL integrates the squared drift error against dt, but batches are
drawn log-uniformly in beta, which under-weights late time by a factor of order
beta.  Turn this on when you care about sample quality rather than uniform
operator accuracy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from . import checkpointing as _ckpt
from .data import BatchSampler, holdout_loss as _holdout_loss, load_dataset
from .models import MLPConfig, TiltMLP
from .noising import build_query, dt_dlogbeta


def train(datadir: str | Path, outpath: str | Path, cfg: MLPConfig | None = None,
          steps: int = 3000, batch_size: int = 1024, lr: float = 1e-3,
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
    """
    eval_fn, if given, is called as eval_fn(net) -> dict[str, float] on a schedule
    (every eval_every steps, defaulting to log_every) with the net in eval mode and
    torch.no_grad().  Results are appended to test_history as {"step": step, **metrics}
    -- e.g. a dict keyed by discrete lambda level, tau, or beta to track how different
    slices of the operator converge at different rates.  Stored in the checkpoint's
    extra["test_history"] and also attached to the returned net as net.test_history.

    holdout, if given (a dict from data.build_frozen_holdout/load_frozen_holdout),
    tracks a ground-truth-free regression loss E||X - G_theta(v,a*)||^2 on a FIXED
    batch drawn once from a source disjoint from BatchSampler's own training-x pool
    -- unlike eval_fn (which needs an exact operator and so is unavailable for
    large/non-enumerable targets), this works for any dataset. Checked every
    holdout_every steps (defaulting to eval_every, then log_every), appended to
    holdout_history as {"step": step, "holdout_mean_sse": ...}, stored in the
    checkpoint's extra["holdout_history"], and logged to wandb under
    "test-holdout/mean_sse" when use_wandb=True. This is the direct mitigation for
    the overfitting failure mode diagnosed on data/sudoku6_rn0034_v1 (BatchSampler
    resampling a fixed, finite train.csv row pool with replacement):
    holdout gives a way to see that failure mode as it happens during training,
    without needing an exact operator, rather than only after the fact by comparing
    two full training runs' final checkpoints.

    use_wandb=True logs the same quantities (train loss, eval_fn metrics) to a
    Weights & Biases run in real time, in addition to the in-memory history/
    test_history lists that always get saved into the checkpoint.

    device selects where the model lives and the training loop runs ("cuda",
    "cpu", a torch.device, or None to auto-pick cuda if available). The
    dataset/batches are generated on CPU regardless (BatchSampler uses a CPU
    generator) and moved to `device` per-batch, since generation is cheap
    relative to the MLP forward/backward pass. eval_fn (from evaluate.py)
    builds its own tensors internally and assumes CPU, so the net is moved
    back to CPU for the duration of each eval_fn call and moved back to
    `device` afterward -- nn.Module.to() mutates parameter .data in place,
    so this round-trip does not disturb the Adam optimizer's state (which is
    keyed by parameter object identity).
    
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

    # n_extra: width of the reward block past the token/core bits. Same
    # convention as train_selfcond.py -- MLPConfig has no n_extra, hence the
    # getattr default of 1, which reproduces every existing run bit-for-bit.
    n_extra = int(getattr(cfg, "n_extra", 1))
    batches = BatchSampler(x, meta["beta_lo"], meta["beta_hi"], frozen,
                           lambda_max=lambda_max, mode1_prob=mode1_prob,
                           lambda1_prob=lambda1_prob, beta_focus_range=beta_focus_range,
                           beta_focus_prob=beta_focus_prob, seed=seed,
                           n_extra=n_extra)
    net = (net if net is not None else TiltMLP(cfg)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    eval_every = eval_every or log_every
    holdout_every = holdout_every or eval_every
    if holdout is not None:
        holdout = {k: (v.to(device) if v is not None else None) for k, v in holdout.items()}
    history, test_history, holdout_history, t0 = [], [], [], time.time()

    # ckpt_every=None reproduces the pre-existing behaviour exactly: no partial
    # writes, no resume, one save after the loop. resume=True on a first
    # submission is a no-op (there is no partial yet), so the SAME sbatch line
    # works for both a fresh run and a requeue after a walltime kill.
    start_step = 1
    if resume:
        start_step, restored = _ckpt.load_partial(outpath, net, opt, sched,
                                                  batches, verbose=verbose)
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
        per = ((xb - net(vb, ab_star)) ** 2).sum(1)
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
            history.append({"step": step, "loss": float(loss)})
            if verbose:
                print(f"  step {step:>6}  loss {float(loss):.4f}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
            if wandb_run is not None:
                wandb_run.log({"train/loss": float(loss), "lr": sched.get_last_lr()[0]},
                               step=step)
        if holdout is not None and (step % holdout_every == 0 or step == steps):
            net.eval()
            hl = _holdout_loss(net, holdout, d=cfg.d, self_cond=False)
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
                               meta={"datadir": str(datadir), "steps": steps,
                                     "seed": seed})

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    net.save(outpath, extra={
        "datadir": str(datadir), "steps": steps, "batch_size": batch_size,
        "lr": lr, "dt_weight": dt_weight, "lambda_max": lambda_max,
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


def _flatten_metrics(metrics: dict, prefix: str = "") -> dict:
    """Flatten eval_fn() output into scalar leaves suitable for wandb.log.

    Handles the shapes evaluate.py functions actually return: a plain dict of
    scalars, a dict keyed by e.g. beta/lambda level with nested dicts, or a
    dict whose values are lists of per-level dicts (each expected to carry an
    identifying key like "beta", "a", or "tau").
    """
    flat = {}
    for key, val in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(val, dict):
            flat.update(_flatten_metrics(val, prefix=f"{name}/"))
        elif isinstance(val, (list, tuple)):
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    ident = item.get("beta", item.get("a", item.get("tau", i)))
                    for k, v in item.items():
                        if k in ("beta", "a", "tau"):
                            continue
                        if isinstance(v, (int, float)):
                            flat[f"{name}/{ident}/{k}"] = v
                elif isinstance(item, (int, float)):
                    flat[f"{name}/{i}"] = item
        elif isinstance(val, (int, float)):
            flat[name] = val
        # non-numeric scalars (e.g. strings) are dropped silently
    return flat


def _sectioned_test_log(flat: dict) -> dict:
    """Bucket flattened eval metrics into separate wandb sections by family.

    Without this, every eval metric (per-beta SSE/RMSE, per-beta Jacobian
    symmetry, per-beta Jacobian diagonal, ...) lands under one "test/" prefix
    -- wandb groups its workspace by the first "/" segment, so that dumps
    everything into a single overcrowded "test" section. Instead, route the
    per-beta SSE/RMSE breakdown (keys like "beta_0.02/sse" or "mean_sse")
    into a shared "test-sse" section, and give every other top-level metric
    family (e.g. "jac_symmetry_random", "jac_diagonal") its own
    "test-<family>" section automatically -- no changes needed here when a
    new eval_fn metric family is added later.
    """
    out = {}
    for k, v in flat.items():
        head = k.split("/", 1)[0]
        section = "sse" if head.startswith("beta_") or head == "mean_sse" else head
        if section == head:
            rest = k[len(head):].lstrip("/")
            out[f"test-{section}/{rest}" if rest else f"test-{section}/{section}"] = v
        else:
            out[f"test-{section}/{k}"] = v
    return out
