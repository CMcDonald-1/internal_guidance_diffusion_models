"""Self-conditioned variant of the tilt-operator network (Chen/Zhang/Hinton, "Analog
Bits", Algorithm 1 training + Algorithm 2 default sampling).

This module is a standalone parallel to models.py -- it does not modify TiltMLP, and
existing checkpoints/training/sampling/eval code for TiltMLP are completely unaffected.
It reuses MLPConfig, LevelEmbedding, ResBlock, and (when cfg.grouped) GroupEncoder from
models.py unchanged (imported, not copied) since those pieces (the FiLM residual stack,
the level embedding, the shared per-slot block encoder) have nothing to do with
self-conditioning.

    G_theta(u, a, x_pred) = tanh( beta(a) * u  +  gain(a) * residual_stack(feat(u), feat(x_pred), a) )

The only architectural change from TiltMLP is that lin_in also takes a second copy of the
input slot, x_pred -- the network's own previous estimate of the target (analogous to the
paper's `x_tilde_0`) -- concatenated alongside u. x_pred defaults to zero, which recovers
(up to the doubled input width / an all-zero block of weights) plain non-self-conditioned
behavior. The mean-field skip connection is built from beta(a)*u only, exactly as in
TiltMLP -- x_pred only ever informs the learned residual.

Grouped featuriser (cfg.grouped, i.e. n_groups/group_bits set). Mirrors TiltMLP's
flat-vs-grouped split exactly, but since there are now TWO input slots (u and x_pred)
that share the same K-groups-of-M-bits layout, each gets its OWN GroupEncoder instance
(encoder_u, encoder_xpred) -- same architecture/shapes, independent weights. Sharing one
encoder's weights across both roles was considered and rejected: u is a raw noisy
observation and x_pred is the network's own near-{-1,1} running estimate, so the two have
different statistics and there is no reason to force one shared encoding for both. Within
a role, the K slots still share one encoder, exactly as in TiltMLP -- that is the part of
the design GroupEncoder exists to enforce. cfg.grouped=False (the default) reproduces the
pre-grouping behavior bit for bit: encoder_u = encoder_xpred = None, feat(u) = u.

Two ways to get a prediction out of a trained net, at different costs:

  forward(u, a, x_pred)   one network pass with an EXPLICIT x_pred (zeros if None) --
                          the raw building block. Training's coin flip and the sampler's
                          step-to-step self-conditioning both call this directly.
  self_cond_predict(u, a) the "deployed" two-pass estimate: first computes
                          x0 = forward(u, a, x_pred=None) with a stop-gradient, then
                          returns forward(u, a, x_pred=x0). This is what a single
                          (non-sequential) query should use to get the network's best
                          single-shot estimate -- see evaluate_selfcond.py.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from .models import GroupEncoder, LevelEmbedding, MLPConfig, ResBlock
from .noising import beta_of_a, normalize_natural


class SelfCondTiltMLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.level = LevelEmbedding(cfg)
        self.encoder_u = GroupEncoder(cfg) if cfg.grouped else None
        self.encoder_xpred = GroupEncoder(cfg) if cfg.grouped else None
        # feat_dim(u) + feat_dim(x_pred) + the level embedding. feat_dim == d when
        # ungrouped, matching the original 2*cfg.d + cfg.n_level exactly.
        self.lin_in = nn.Linear(2 * cfg.feat_dim + cfg.n_level, cfg.width)
        self.blocks = nn.ModuleList([ResBlock(cfg) for _ in range(cfg.n_blocks)])
        self.norm_out = nn.LayerNorm(cfg.width)
        self.lin_out = nn.Linear(cfg.width, cfg.d)
        self.gain = nn.Sequential(
            nn.Linear(cfg.n_level, cfg.gain_hidden), nn.SiLU(),
            nn.Linear(cfg.gain_hidden, cfg.d))
        if cfg.skip:
            nn.init.zeros_(self.lin_out.weight)
            nn.init.zeros_(self.lin_out.bias)

    def forward(self, u: torch.Tensor, a: torch.Tensor,
                x_pred: torch.Tensor | None = None) -> torch.Tensor:
        """One network pass given an explicit (possibly zero) self-conditioning input.

        x_pred=None is treated as zeros_like(u) -- the "no prior estimate" case (the
        very first sampling step, or the non-self-cond half of training's coin flip).
        """
        if x_pred is None:
            x_pred = torch.zeros_like(u)
        if a.dim() == 1:
            a = a.unsqueeze(1)
        if u.shape[-1] != self.cfg.d:
            raise ValueError(f"expected u of width {self.cfg.d}, got {u.shape[-1]}")
        if x_pred.shape[-1] != self.cfg.d:
            raise ValueError(f"expected x_pred of width {self.cfg.d}, got {x_pred.shape[-1]}")
        e = self.level(a)
        xu = u / math.sqrt(self.cfg.d)
        xp = x_pred / math.sqrt(self.cfg.d)
        if self.encoder_u is not None:
            xu = self.encoder_u(xu)
            xp = self.encoder_xpred(xp)
        h = self.lin_in(torch.cat([xu, xp, e], dim=1))
        for blk in self.blocks:
            h = blk(h, e)
        r = self.lin_out(self.norm_out(h))
        if self.cfg.skip:
            # raw u, not the scaled/grouped features: the mean-field skip is
            # identical in the flat and grouped variants (matches TiltMLP).
            return torch.tanh(beta_of_a(a) * u + self.gain(e) * r)
        return torch.tanh(r)

    def self_cond_predict(self, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """The two-pass "deployed" estimate: f(u, f(u, 0, a).detach(), a).

        This is the single-shot query self-conditioning is meant to improve on the
        plain x_pred=0 pass -- use this (not forward(u, a)) whenever there's no real
        previous estimate to carry over (e.g. one-off evaluation queries). The
        sampler instead carries the previous SDE step's own estimate step-to-step
        via forward(..., x_pred=...) directly, which is cheaper (one pass per step).
        """
        with torch.no_grad():
            x0 = self.forward(u, a, x_pred=None)
        return self.forward(u, a, x_pred=x0)

    def natural(self, h: torch.Tensor, x_pred: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate at a natural parameter h, via the normalised parameterisation.

        Cheap single-pass hook (mirrors TiltMLP.natural) -- the sampler uses this,
        supplying its own running x_pred estimate rather than paying for a second
        forward pass at every SDE step. x_pred=None (the default) gives the plain
        x_pred=0 pass, matching TiltMLP.natural's behavior when self-conditioning
        provides no information.
        """
        v, a = normalize_natural(h, self.cfg.d)
        return self.forward(v, a, x_pred=x_pred)

    def natural_self_cond(self, h: torch.Tensor) -> torch.Tensor:
        """Two-pass "deployed" estimate at a natural parameter h (see self_cond_predict)."""
        v, a = normalize_natural(h, self.cfg.d)
        return self.self_cond_predict(v, a)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- io ---------------------------------------------------------------
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        torch.save({"cfg": asdict(self.cfg), "state_dict": self.state_dict(),
                    "extra": extra or {}}, path)

    @staticmethod
    def load(path: str | Path, map_location="cpu"):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        cfg = MLPConfig.from_dict(ck["cfg"])
        net = SelfCondTiltMLP(cfg)
        net.load_state_dict(ck["state_dict"])
        net.eval()
        return net, ck.get("extra", {})
