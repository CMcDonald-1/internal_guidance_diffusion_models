"""Opt-in bf16 autocast, as a net wrapper rather than a change to train().

WHY A WRAPPER
-------------
`tilt/train.py` and `tilt/train_selfcond.py` are shared by every run in this
project, including four sudoku6 arms and the whole rhyme 2x2 sweep. Adding an
`amp=` branch inside those loops would put a new conditional in the hot path of
runs that have already been used to draw conclusions. `train(net=...)` already
exists as the hook for an alternative featuriser, and the loop "only ever calls
the net, never inspects its class" -- so autocast can live entirely in the net.
Both training loops stay byte-for-byte unchanged, and a run that does not ask
for AMP cannot be affected by this file at all.

WHY IT IS NEEDED FOR v3
-----------------------
Training is fp32 today -- there is no autocast anywhere in the project. The v3
transformer is 33 attention slots (32 tokens + the reward coord) against v2's
11, at 12 layers against 6. At batch 8192 the feed-forward hidden activation
alone is 8192 x 33 x 2048 fp32 = 2.2 GB PER BLOCK, and there are twelve of them.
That does not fit in an H100-80 alongside everything else.

bf16 autocast halves activation memory and buys roughly 1.5-2x on H100 tensor
cores, which is also what pulls a ~34-48h estimated run back under a 48h
cluster GPU walltime ceiling -- a ceiling that matters more than usual here
because `train()` has no resume path (the cosine schedule uses `steps` as
T_max, Adam state is never checkpointed, and `--ckpt-every` writes weights
only). A walltime kill loses the run.

bf16 rather than fp16: same exponent range as fp32, so no GradScaler and no
overflow bookkeeping. Parameters and the optimizer stay fp32; only the forward
activations are reduced, and the output is cast back to fp32 so the regression
loss is accumulated at full precision.

WHAT TO CHECK BEFORE TRUSTING IT
--------------------------------
This is a numerics change relative to every existing baseline. The probe
(scripts/probe_xf_v3.py) runs matched fp32 and bf16 configurations and reports
both loss curves; they should track within noise. On this problem the loss is
~97% Bayes floor and moved only 4 nats across an entire 100k run, so a
divergence of even a nat is meaningful and worth stopping for.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None,
           "none": None}


class AutocastNet(nn.Module):
    """Wraps a tilt net so every forward runs under autocast, output in fp32.

    Delegates the full surface `train()`, `train_selfcond()`, `sl_sample` and
    the evaluation scripts use: `forward(u, a, x_pred=None)`, `.natural`,
    `.natural_self_cond`, `.self_cond_predict`, `.save`, `.n_params`, `.cfg`.
    Parameters live on the inner module, so `.parameters()`, `.to()`, `.train()`
    and `.eval()` work through nn.Module's own machinery.

    `.save()` delegates to the inner net, so the checkpoint on disk is an
    ORDINARY TransformerTilt checkpoint -- `TransformerTilt.load` reads it with
    no knowledge of this class, and nothing downstream (memcheck, repair
    distance, sample-quality sweep) needs to change. AMP is a training-time
    detail and deliberately leaves no trace in the artifact.
    """

    def __init__(self, net: nn.Module, dtype: str | torch.dtype = "bf16"):
        super().__init__()
        if isinstance(dtype, str):
            if dtype not in _DTYPES:
                raise ValueError(f"amp dtype must be one of {sorted(_DTYPES)}, "
                                 f"got {dtype!r}")
            dtype = _DTYPES[dtype]
        if dtype is None:
            raise ValueError("AutocastNet built with amp disabled -- callers "
                             "should use the bare net instead of wrapping it")
        self.inner = net
        self.amp_dtype = dtype

    # -- plumbing ------------------------------------------------------------
    @property
    def cfg(self):
        return self.inner.cfg

    def _ctx(self):
        # CPU autocast exists but is a pessimisation here, and the smoke tests
        # run on CPU. Fall through to a no-op rather than silently changing
        # numerics on a machine where there is nothing to gain.
        dev = next(self.inner.parameters()).device.type
        if dev != "cuda":
            return torch.autocast(device_type=dev, enabled=False)
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    # -- forward paths -------------------------------------------------------
    def forward(self, u, a, x_pred=None):
        with self._ctx():
            out = self.inner(u, a, x_pred)
        return out.float()

    def natural(self, h, x_pred=None):
        with self._ctx():
            out = self.inner.natural(h, x_pred)
        return out.float()

    def self_cond_predict(self, u, a):
        with self._ctx():
            out = self.inner.self_cond_predict(u, a)
        return out.float()

    def natural_self_cond(self, h):
        with self._ctx():
            out = self.inner.natural_self_cond(h)
        return out.float()

    # -- pass-through --------------------------------------------------------
    def n_params(self) -> int:
        return self.inner.n_params()

    def save(self, path, extra=None):
        extra = dict(extra or {})
        extra["amp"] = str(self.amp_dtype).replace("torch.", "")
        return self.inner.save(path, extra=extra)


def maybe_autocast(net: nn.Module, amp: str | None):
    """Return `net` unchanged when amp is off. The one call site's whole API."""
    if not amp or amp in ("off", "none"):
        return net
    return AutocastNet(net, amp)


if __name__ == "__main__":   # python -m tilt.amp
    from .models_transformer import TransformerTilt, transformer_config

    cfg = transformer_config(32, 15, d_model=128, n_layers=2, n_heads=4,
                             self_cond=True)
    net = TransformerTilt(cfg)
    wrapped = maybe_autocast(net, "bf16")
    u, a = torch.randn(4, cfg.d), torch.rand(4)

    assert maybe_autocast(net, None) is net
    assert maybe_autocast(net, "off") is net
    # Same parameter objects, so Adam state and .to(device) behave identically.
    assert list(wrapped.parameters()) and \
        all(p is q for p, q in zip(wrapped.parameters(), net.parameters()))
    assert wrapped.cfg is cfg and wrapped.n_params() == net.n_params()
    # On CPU the wrapper is a no-op, so it must reproduce the bare net exactly.
    assert torch.allclose(wrapped(u, a), net(u, a), atol=0)
    assert torch.allclose(wrapped.self_cond_predict(u, a),
                          net.self_cond_predict(u, a), atol=0)
    assert wrapped(u, a).dtype is torch.float32
    print(f"tilt.amp ok  ({net.n_params():,} params, wrapper is transparent on CPU)")
