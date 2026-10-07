"""Transformer featuriser for the tilt-mean operator.

WHY
---
The flat `TiltMLP` and the grouped `GroupEncoder` both give the network at most
SLOT IDENTITY: the grouped variant applies one shared 15->h_group linear to each
token block plus per-slot FiLM, but the blocks only ever meet inside the flat
trunk. Nothing lets token j's bits attend to token k's.

That is the exact failure observed in an earlier text experiment: samples decode
to sequences of individually valid BPE tokens that cannot compose into words
(`GotPAyl`, `Unasons`, `somening`), while position-0 structure is learned well
(`KANSAS CITY, Mo.`, `The U.S.`). An autoregressive model structurally cannot
emit those because every token is conditioned on the prefix; a flat joint model
over 150 bits has no such constraint and must discover the factorisation from
data alone.

Chen, Zhang & Hinton (Analog Bits, ICLR 2023) use exactly this project's
representation for text -- sentencepiece 32K vocab, "we encode each token into
15 analog bits" (p.4) -- and model it with "a randomly initialized 6-layer
Transformer decoder with 512 dimension per layer... non-masked all-to-all
attention" (p.5). No MLP appears anywhere in that paper. This module is that
architecture, adapted to the (v*, s*) parameterisation of the tilt write-up.

WHAT IS PRESERVED
-----------------
1. THE INTERFACE. `natural(h)`, `.cfg`, `.save()`, `.load()`, and a
   `forward(u, a)` returning (B, d) in (-1, 1). `sl_sample` takes `net.natural`
   as a plain callable, so the sampler, memorization check and log-likelihood
   scripts work unchanged.

2. THE MEAN-FIELD SKIP, hence both limits. `TiltMLP` returns
   `tanh(beta(a) * u + gain(e) * r)` with a zero-initialised output layer, so at
   init the network IS the mean-field denoiser exactly, beta -> infinity gives
   the identity map and beta -> 0 gives E_pi[X]. `MLPConfig` annotates
   `group_depth=1` as "keeps the exactness result". Those limits are load-bearing
   at both ends of the sampler trajectory, so the same head is used verbatim here
   and every residual branch is zero-initialised. The __main__ block asserts a
   fresh net reproduces `tanh(beta(a) * u)` to 1e-6, exactly as tilt/models.py
   does for TiltMLP.

3. THE ONE-OPERATOR PROPERTY. The write-up's claim -- "any distribution over
   (q, lambda, a) leaves the pointwise minimizer at E_pi[X|eta]" -- is a fact
   about the regression loss and its target, not about the function class. Any
   g_theta(v*, s*) -> R^d inherits it.

DESIGN NOTES
------------
* The reward coordinate is its OWN TOKEN, not folded into the level
  conditioning. The whole construction is about tilting along e_*, so attention
  should be able to relate x_* to every token directly and vice versa. With
  seq_len=10 that makes an 11-token sequence.
* Level conditioning is AdaLN (DiT-style) from the existing `LevelEmbedding`
  Fourier features, so `level_on="logbeta"` remains available.
* Self-conditioning follows Analog Bits exactly: the previous x_0 estimate is
  concatenated to the input ALONG THE FEATURE AXIS PER TOKEN (`cat([x_crpt,
  x_pred], -1)`, p.4), i.e. 15 -> 30 channels per slot, not as extra positions.
* Attention is non-causal. The denoiser is not autoregressive.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import LevelEmbedding
from .noising import beta_of_a, normalize_natural


@dataclass
class TransformerConfig:
    """Mirrors MLPConfig's shared fields so runners can stay uniform."""

    # d is the FULL input width including the reward coordinate:
    # seq_len * token_bits + n_extra.  10 x 15 + 1 = 151 for rhyme_gpt2.
    d: int = 151
    seq_len: int = 10
    token_bits: int = 15

    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int | None = None          # defaults to 4 * d_model
    dropout: float = 0.0

    self_cond: bool = False          # concatenate x_pred per token (Analog Bits)

    # -- shared with MLPConfig, same semantics ---------------------------------
    skip: bool = True
    fourier: tuple = (1.0, 2.0, 4.0, 8.0)
    gain_hidden: int = 32
    level_on: str = "a"              # "a" | "logbeta"

    @property
    def n_level(self) -> int:
        return 2 + 2 * len(self.fourier)

    @property
    def n_extra(self) -> int:
        """Coordinates past the token blocks -- the reward coord (1 for d=151)."""
        return self.d - self.seq_len * self.token_bits

    @property
    def n_slots(self) -> int:
        """Sequence length seen by attention: tokens plus the reward slot."""
        return self.seq_len + (1 if self.n_extra > 0 else 0)

    @property
    def ff_dim(self) -> int:
        return self.d_ff or 4 * self.d_model

    @property
    def tag(self) -> str:
        sc = "_sc" if self.self_cond else ""
        return (f"xf{self.seq_len}x{self.token_bits}_d{self.d_model}"
                f"x{self.n_layers}h{self.n_heads}{sc}_d{self.d}")

    def validate(self) -> "TransformerConfig":
        if self.level_on not in ("a", "logbeta"):
            raise ValueError(f"level_on must be 'a' or 'logbeta', got {self.level_on!r}")
        if self.seq_len < 1 or self.token_bits < 1:
            raise ValueError("seq_len and token_bits must be >= 1")
        if self.seq_len * self.token_bits > self.d:
            raise ValueError(
                f"seq_len*token_bits = {self.seq_len * self.token_bits} exceeds d = {self.d}")
        if self.n_extra < 0:
            raise ValueError(f"negative n_extra ({self.n_extra})")
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model={self.d_model} not divisible by n_heads={self.n_heads}")
        return self

    @staticmethod
    def from_dict(dd: dict) -> "TransformerConfig":
        known = {f.name for f in fields(TransformerConfig)}
        out = {k: v for k, v in dd.items() if k in known}
        if "fourier" in out:
            out["fourier"] = tuple(out["fourier"])
        return TransformerConfig(**out).validate()


class AdaLNBlock(nn.Module):
    """Pre-norm transformer block, AdaLN-conditioned on the noise level.

    Both residual branches are gated by a zero-initialised scale, so a fresh
    block is EXACTLY the identity. That is what lets the whole network start at
    the mean-field denoiser, matching ResBlock's zero-init convention in
    tilt/models.py.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        w = cfg.d_model
        self.norm1 = nn.LayerNorm(w, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(w, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(w, cfg.n_heads, dropout=cfg.dropout,
                                          batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(w, cfg.ff_dim), nn.SiLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, w))
        # 6 modulation params per block: shift/scale/gate for attn and for ff.
        self.mod = nn.Linear(cfg.n_level, 6 * w)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        # e: (B, n_level) -> six (B, 1, d_model) modulators broadcast over slots.
        sh1, sc1, g1, sh2, sc2, g2 = self.mod(e).unsqueeze(1).chunk(6, dim=-1)
        x = (1.0 + sc1) * self.norm1(h) + sh1
        h = h + g1 * self.attn(x, x, x, need_weights=False)[0]
        y = (1.0 + sc2) * self.norm2(h) + sh2
        return h + g2 * self.ff(y)


class TransformerTilt(nn.Module):
    """Attention over token slots, with TiltMLP's mean-field skip head."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg = cfg.validate()
        M, W = cfg.token_bits, cfg.d_model

        in_tok = M * (2 if cfg.self_cond else 1)
        in_ext = max(cfg.n_extra, 1) * (2 if cfg.self_cond else 1)
        self.tok_in = nn.Linear(in_tok, W)
        self.ext_in = nn.Linear(in_ext, W) if cfg.n_extra > 0 else None
        self.pos = nn.Parameter(torch.zeros(cfg.n_slots, W))
        nn.init.normal_(self.pos, std=0.02)

        self.level = LevelEmbedding(cfg)
        self.blocks = nn.ModuleList([AdaLNBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_out = nn.LayerNorm(W, elementwise_affine=False)
        self.mod_out = nn.Linear(cfg.n_level, 2 * W)
        self.tok_out = nn.Linear(W, M)
        self.ext_out = nn.Linear(W, cfg.n_extra) if cfg.n_extra > 0 else None
        self.gain = nn.Sequential(
            nn.Linear(cfg.n_level, cfg.gain_hidden), nn.SiLU(),
            nn.Linear(cfg.gain_hidden, cfg.d))

        # Zero-init every path that could perturb the mean-field skip, so a fresh
        # net returns tanh(beta(a) * u) exactly -- same convention as TiltMLP.
        for m in (self.mod_out, self.tok_out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        if self.ext_out is not None:
            nn.init.zeros_(self.ext_out.weight); nn.init.zeros_(self.ext_out.bias)

    # -- core ---------------------------------------------------------------
    def _trunk(self, u: torch.Tensor, e: torch.Tensor,
               x_pred: torch.Tensor | None) -> torch.Tensor:
        """(B, d) raw residual r, before the skip head."""
        cfg = self.cfg
        B, K, M = u.shape[0], cfg.seq_len, cfg.token_bits
        blocks = u[:, : K * M].view(B, K, M)
        if cfg.self_cond:
            xp = torch.zeros_like(u) if x_pred is None else x_pred
            blocks = torch.cat([blocks, xp[:, : K * M].view(B, K, M)], dim=-1)
        h = self.tok_in(blocks)                                    # (B, K, W)

        if self.ext_in is not None:
            ext = u[:, K * M:]
            if cfg.self_cond:
                xp = torch.zeros_like(u) if x_pred is None else x_pred
                ext = torch.cat([ext, xp[:, K * M:]], dim=-1)
            h = torch.cat([h, self.ext_in(ext).unsqueeze(1)], dim=1)  # (B, K+1, W)

        h = h + self.pos
        for blk in self.blocks:
            h = blk(h, e)
        sh, sc = self.mod_out(e).unsqueeze(1).chunk(2, dim=-1)
        h = (1.0 + sc) * self.norm_out(h) + sh

        r = self.tok_out(h[:, :K]).reshape(B, K * M)
        if self.ext_out is not None:
            r = torch.cat([r, self.ext_out(h[:, K])], dim=1)
        return r

    def forward(self, u: torch.Tensor, a: torch.Tensor,
                x_pred: torch.Tensor | None = None) -> torch.Tensor:
        if a.dim() == 1:
            a = a.unsqueeze(1)
        if u.shape[-1] != self.cfg.d:
            raise ValueError(f"expected u of width {self.cfg.d}, got {u.shape[-1]}")
        e = self.level(a)
        r = self._trunk(u, e, x_pred)
        if self.cfg.skip:
            return torch.tanh(beta_of_a(a) * u + self.gain(e) * r)
        return torch.tanh(r)

    # -- sampler entry points ------------------------------------------------
    def natural(self, h: torch.Tensor, x_pred: torch.Tensor | None = None
                ) -> torch.Tensor:
        """Evaluate at a natural parameter h. The entry point sl_sample uses."""
        v, a = normalize_natural(h, self.cfg.d)
        return self.forward(v, a, x_pred)

    def self_cond_predict(self, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Two-pass deployed estimate: predict with x_pred=0, then feed it back.

        Mirrors SelfCondTiltMLP.self_cond_predict and Analog Bits Algorithm 1 --
        the first pass is detached, so this costs one extra forward and no extra
        backward.
        """
        if not self.cfg.self_cond:
            return self.forward(u, a)
        with torch.no_grad():
            first = self.forward(u, a, None)
        return self.forward(u, a, first.detach())

    def natural_self_cond(self, h: torch.Tensor) -> torch.Tensor:
        v, a = normalize_natural(h, self.cfg.d)
        return self.self_cond_predict(v, a)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- io ------------------------------------------------------------------
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        torch.save({"cfg": asdict(self.cfg), "arch": "transformer",
                    "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @staticmethod
    def load(path: str | Path, map_location="cpu"):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        net = TransformerTilt(TransformerConfig.from_dict(ck["cfg"]))
        net.load_state_dict(ck["state_dict"])
        net.eval()
        return net, ck.get("extra", {})


def transformer_config(seq_len: int, token_bits: int, reward: bool = True,
                       **kw) -> TransformerConfig:
    """Mirrors models.grouped_config: K slots of M bits, plus a reward coord.

    GPT-2, 10 tokens of 15 bits:  transformer_config(10, 15) -> d = 151
    """
    d = seq_len * token_bits + (1 if reward else 0)
    return TransformerConfig(d=d, seq_len=seq_len, token_bits=token_bits,
                             **kw).validate()


if __name__ == "__main__":  # python -m tilt.models_transformer
    torch.manual_seed(0)
    cases = {
        "gpt2 xf":        transformer_config(10, 15, d_model=512, n_layers=6),
        "gpt2 xf small":  transformer_config(10, 15, d_model=384, n_layers=4,
                                             n_heads=6),
        "gpt2 xf selfc":  transformer_config(10, 15, d_model=512, n_layers=6,
                                             self_cond=True),
        "sudoku6 by row": transformer_config(6, 18, d_model=256, n_layers=4),
    }
    for name, cfg in cases.items():
        net = TransformerTilt(cfg)
        u = torch.randn(7, cfg.d)
        a = torch.rand(7)
        y = net(u, a)
        assert y.shape == (7, cfg.d), y.shape
        assert y.abs().max() <= 1.0
        # zero-init residual => a fresh net IS the mean-field denoiser, exactly.
        assert torch.allclose(y, torch.tanh(beta_of_a(a[:, None]) * u), atol=1e-6), \
            f"{name}: fresh net departs from mean field"
        print(f"{name:16s} {cfg.tag:44s} {net.n_params():>10,} params")
