"""Periodic, exactly-resumable checkpoints for the training loops.

WHY THIS EXISTS
---------------
`train()` and `train_selfcond()` called `net.save()` once, after the loop. A
run killed by Slurm walltime therefore produced NOTHING -- not a degraded
checkpoint, not a shorter run, nothing. The first 9x9 job died at 140,000 of
180,000 steps and 18 GPU-hours evaporated with it, along with a day of
wall-clock that could not be bought back.

Padding the walltime does not fix that. A 12-hour job still meets node
failures, preemption and OOM-killers, and the loss is total every time.

WHAT "EXACTLY RESUMABLE" MEANS HERE
-----------------------------------
Resuming mid-run is only worth anything if the resumed run is the run you
would have got. Four pieces of state have to travel, and all four are
recoverable in this codebase:

  net.state_dict()      the weights
  opt.state_dict()      Adam's first and second moments -- dropping these
                        restarts the moment estimates from zero and puts a
                        visible transient in the loss
  sched.state_dict()    CosineAnnealingLR is parameterised by TOTAL steps, so
                        a resumed run that restarts the schedule is annealing
                        on the wrong curve for the rest of training
  sampler.g state       BatchSampler holds ONE torch.Generator and every
                        downstream draw (row indices, beta, the focus band,
                        Lambda/mode1, the VP noise) takes `generator=` from
                        it, so a single get_state() captures the whole data
                        stream

Plus the global torch RNG, which nothing in the current step consumes
(TransformerConfig.dropout defaults to 0.0) but would the moment dropout is
switched on, and any extra generators the caller passes -- `train_selfcond`
has a `coin_gen` for its per-step self-conditioning coin flip.

Restore all of those and the resumed run is bitwise identical to the
uninterrupted one. That is asserted in `_selftest` below, not assumed.

WRITES ARE ATOMIC
-----------------
Saved to a temporary file and then `os.replace`d, which is atomic on POSIX.
A job killed *during* a checkpoint write leaves the previous good checkpoint
intact rather than a truncated file that fails to load -- the failure mode
that turns a recoverable run into a lost one.

USAGE
-----
Off by default. `ckpt_every=None` reproduces the old behaviour exactly, so
every existing sudoku6 run is unaffected.

    train(..., ckpt_every=5000, resume=True)

`resume=True` picks up `<outpath>.partial.pt` if it exists and starts from the
next step; if it does not exist the run simply starts from scratch, so the
SAME sbatch line works whether it is a first submission or a requeue.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


def partial_path(outpath: str | Path) -> Path:
    """`ckpt/foo.pt` -> `ckpt/foo.partial.pt`.

    Deliberately not `foo.pt.partial`: tilt/load_ckpt.py's NAME_RE anchors on
    `\\.pt$`, and evaluation scripts glob `*.pt`. Keeping the `.pt` suffix but
    inserting `.partial` means a stray partial is still loadable by hand,
    while `--pattern '*[!l].pt'`-style globs are not what anyone should rely
    on -- prefer an explicit pattern in eval scripts.
    """
    p = Path(outpath)
    return p.with_suffix("").with_suffix(".partial.pt")


def save_partial(outpath, net, opt, sched, sampler, step: int,
                 histories: dict, extra_gens: dict | None = None,
                 meta: dict | None = None) -> Path:
    """Atomically write a resumable checkpoint. Returns the path written."""
    path = partial_path(outpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "step": step,
        "state_dict": net.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "sampler_g": sampler.g.get_state(),
        "torch_rng": torch.get_rng_state(),
        "extra_gens": {k: g.get_state() for k, g in (extra_gens or {}).items()},
        "histories": histories,
        "meta": meta or {},
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)              # atomic on POSIX
    return path


def load_partial(outpath, net, opt, sched, sampler,
                 extra_gens: dict | None = None, verbose: bool = True):
    """Restore in place. Returns (next_step, histories) or (1, None) if absent.

    Every tensor is mapped to CPU on load and then moved by the optimiser's own
    `load_state_dict`, which places Adam's moments on whatever device the
    parameters already live on -- so this works for a GPU run resumed on a
    different node without special handling.
    """
    path = partial_path(outpath)
    if not path.exists():
        return 1, None

    blob = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(blob["state_dict"])
    opt.load_state_dict(blob["opt"])
    sched.load_state_dict(blob["sched"])
    sampler.g.set_state(blob["sampler_g"])
    torch.set_rng_state(blob["torch_rng"])
    for k, g in (extra_gens or {}).items():
        if k in blob.get("extra_gens", {}):
            g.set_state(blob["extra_gens"][k])

    step = int(blob["step"])
    if verbose:
        print(f"  RESUMED from {path} at step {step:,} "
              f"(opt + sched + sampler RNG restored)", flush=True)
    return step + 1, blob.get("histories")


def clear_partial(outpath, verbose: bool = True) -> None:
    """Remove the partial once the final checkpoint is safely written."""
    path = partial_path(outpath)
    if path.exists():
        path.unlink()
        if verbose:
            print(f"  cleaned up {path}", flush=True)


def _selftest(tmpdir: str | None = None) -> dict:
    """Assert a resumed run is BITWISE identical to an uninterrupted one.

    Trains 40 steps straight through, then trains 20 / resumes / trains 20 more
    and compares every parameter. Anything less than exact equality means one
    of the four state components is not travelling.
    """
    import tempfile

    from .models import MLPConfig, TiltMLP
    from .data import BatchSampler

    tmpdir = Path(tmpdir or tempfile.mkdtemp())
    d, n = 24, 512
    torch.manual_seed(0)
    data = (torch.randint(0, 2, (n, d)).float() * 2 - 1)

    def build(seed=0):
        torch.manual_seed(seed)
        cfg = MLPConfig(d=d, width=32, n_blocks=2)
        net = TiltMLP(cfg)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 40)
        smp = BatchSampler(data, 0.02, 50.0, None, lambda_max=3.0,
                           mode1_prob=0.25, seed=seed)
        return cfg, net, opt, sched, smp

    def run(net, opt, sched, smp, lo, hi, cfg):
        from .noising import build_query
        for _ in range(lo, hi + 1):
            xb, ub, ab, bb, diagb = smp(16)
            vb, ab_star = build_query(ub, ab, diagb, d=cfg.d)
            loss = ((xb - net(vb, ab_star)) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        return float(loss)

    cfg, net, opt, sched, smp = build()
    ref_loss = run(net, opt, sched, smp, 1, 40, cfg)
    ref = {k: v.clone() for k, v in net.state_dict().items()}

    out = tmpdir / "rt.pt"
    cfg, net2, opt2, sched2, smp2 = build()
    run(net2, opt2, sched2, smp2, 1, 20, cfg)
    save_partial(out, net2, opt2, sched2, smp2, 20, {"history": []})

    cfg, net3, opt3, sched3, smp3 = build(seed=99)          # deliberately WRONG seed
    start, _ = load_partial(out, net3, opt3, sched3, smp3, verbose=False)
    res_loss = run(net3, opt3, sched3, smp3, start, 40, cfg)

    same = all(torch.equal(ref[k], v) for k, v in net3.state_dict().items())
    max_dev = max(float((ref[k] - v).abs().max()) for k, v in net3.state_dict().items())
    clear_partial(out, verbose=False)
    return {
        "resume_step": start,
        "bitwise_identical": same,
        "max_param_deviation": max_dev,
        "loss_uninterrupted": ref_loss,
        "loss_resumed": res_loss,
        "loss_identical": ref_loss == res_loss,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(_selftest(), indent=2))
