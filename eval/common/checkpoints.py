"""Shared checkpoint loading + on-device sampling helpers for eval/*.py.

Depends on:
    tilt/load_ckpt.py   (load_tilt_checkpoint, sample_fns -- architecture
                         dispatch: plain TiltMLP vs. SelfCondTiltMLP vs.
                         TransformerTilt, read from the checkpoint's own
                         "arch" field, not guessed from the filename)
"""
from __future__ import annotations

import torch

from tilt.load_ckpt import load_tilt_checkpoint, sample_fns


class OnDevice:
    """Run a net's forward passes on `device` while sampler state stays on CPU.

    tilt/sampler.py and tilt/sampler_selfcond.py hold SDE state on CPU
    (shared across every problem in this repo, deliberately not modified);
    this shim moves only the network evaluation itself. Consolidates what was
    previously one copy-pasted class per eval script (run_sudoku6/9/mnist
    reward-tilt-check and memorization scripts).
    """

    def __init__(self, net, device):
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()

    def _wrap(self, fn):
        def call(u, *args, **kw):
            u = u.to(self.device)
            args = tuple(v.to(self.device) if torch.is_tensor(v) else v for v in args)
            kw = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                  for k, v in kw.items()}
            return fn(u, *args, **kw).detach().cpu()
        return call

    @property
    def natural(self):
        return self._wrap(self.net.natural)

    @property
    def natural_self_cond(self):
        return self._wrap(self.net.natural_self_cond)


def load_premier(path: str, device: str = "cpu"):
    """Load a premier checkpoint for sampling.

    Returns (shim, info, sl_sample_fn, decode_fn, needs_net, drift_arg).

    `drift_arg` is what to pass as the sampler's drift argument: `shim`
    itself for self-conditioned/transformer arms (which need the net object,
    not a bare function), or `shim.natural` for plain arms. Callers should
    always use `drift_arg`, not `shim`, when calling `sl`/`decode`.
    """
    net, info = load_tilt_checkpoint(path)
    shim = OnDevice(net, device)
    sl, decode, needs_net = sample_fns(info)
    drift_arg = shim if needs_net else shim.natural
    return shim, info, sl, decode, needs_net, drift_arg
