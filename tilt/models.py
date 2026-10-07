"""The tilt-operator network.

    G_theta(u, a) = tanh( beta(a) * u  +  gain(a) * residual_stack(feat(u), a) )

The beta(a)*u skip is the exact answer for a uniform pi (mean field), so it
carries the a -> 1 divergence analytically and the residual stack only has to
learn the smooth correlation correction.  tanh enforces the [-1,1]^d box by
construction.  The last linear layer is zero-initialised, so training starts
exactly at the mean-field prediction.

Two input featurisers, selected by config:

  flat   (n_groups=None)  feat(u) = u                      -- current behaviour
  group  (n_groups=K)     feat(u) = [ E(u_1), ..., E(u_K), u_extra ]

where the K groups are contiguous blocks of M coordinates and E is ONE shared
encoder applied to every block.  This is for inputs whose coordinates come in
repeated same-kind blocks: 10 GPT-2 tokens of 15 bits each, or 4 Sudoku rows
of 8 bits each.  The bit->symbol code is identical in every slot, so the map
"what does this bit pattern denote" is worth learning once from K times the
data; what differs by slot is the conditional distribution, and that is left
entirely to lin_in and the residual stack, whose weights are per-slot.

Expressiveness note.  With h_group >= group_bits and a LINEAR encoder, this is
a reparameterisation, not a restriction: the shared W has a left inverse, so
each slot's block of lin_in can realise any per-slot map it wants
(B_i = T_i W^+).  Below group_bits it becomes a rank bottleneck that really
does destroy per-slot information -- validate() refuses that case.  Setting
group_depth=2 adds a nonlinearity and forfeits the guarantee; it is offered as
an experiment, not a default.

Everything downstream of feat(u) -- level embedding, FiLM residual blocks,
gain, skip, tanh, output width -- is untouched by the grouping.  In particular
the skip still acts on the raw u in coordinate space, so the mean-field
initialisation is identical in both variants.

Set skip=False to ablate; empirically that costs little at weak tilt and a lot
at strong tilt, which is the region the Girsanov bound weights most heavily.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .noising import beta_of_a


@dataclass
class MLPConfig:
    # d is the FULL input width, including the reward coordinate if present.
    # e.g. 10 GPT-2 tokens x 15 bits + reward  -> d = 151
    #      4x4 Sudoku (32 bits) + reward       -> d = 33
    d: int = 16
    width: int = 96
    n_blocks: int = 4
    skip: bool = True
    fourier: tuple = (1.0, 2.0, 4.0, 8.0)
    gain_hidden: int = 32

    # -- group ("token level") featuriser -------------------------------------
    # n_groups=None reproduces the flat model bit for bit.
    n_groups: int | None = None      # K
    group_bits: int | None = None    # M.  K*M <= d; the remaining d - K*M
                                     # coordinates (the reward coord) pass raw.
    h_group: int = 32                # per-slot feature width.  MUST be >= M.
    group_cond: str = "film"         # "none" | "bias" | "film"
    group_depth: int = 1             # 1 = linear (keeps the exactness result)

    # -- level embedding carrier ----------------------------------------------
    # "a" is the current behaviour.  "logbeta" spreads the Fourier features
    # evenly in the natural parameter instead of compressing every beta > 5
    # into a in [0.905, 1].  Opt-in so old runs stay reproducible.
    level_on: str = "a"              # "a" | "logbeta"

    @property
    def n_level(self) -> int:
        return 2 + 2 * len(self.fourier)

    @property
    def grouped(self) -> bool:
        return self.n_groups is not None and self.group_bits is not None

    @property
    def n_extra(self) -> int:
        """Coordinates not covered by the groups (typically the reward coord)."""
        return self.d - self.n_groups * self.group_bits if self.grouped else 0

    @property
    def feat_dim(self) -> int:
        """Width handed to lin_in, before the level features are concatenated."""
        if not self.grouped:
            return self.d
        return self.n_groups * self.h_group + self.n_extra

    @property
    def tag(self) -> str:
        """Short identifier for run names."""
        if not self.grouped:
            return f"flat_d{self.d}_w{self.width}x{self.n_blocks}"
        return (f"grp{self.n_groups}x{self.group_bits}_h{self.h_group}"
                f"_{self.group_cond}_d{self.d}_w{self.width}x{self.n_blocks}")

    def validate(self) -> "MLPConfig":
        if self.level_on not in ("a", "logbeta"):
            raise ValueError(f"level_on must be 'a' or 'logbeta', got {self.level_on!r}")
        if (self.n_groups is None) != (self.group_bits is None):
            raise ValueError("set both n_groups and group_bits, or neither")
        if not self.grouped:
            return self
        K, M = self.n_groups, self.group_bits
        if K < 1 or M < 1:
            raise ValueError(f"n_groups={K}, group_bits={M} must both be >= 1")
        if K * M > self.d:
            raise ValueError(f"n_groups*group_bits = {K*M} exceeds d = {self.d}")
        if self.h_group < M:
            raise ValueError(
                f"h_group={self.h_group} < group_bits={M}. The shared encoder "
                f"would be a rank-{self.h_group} bottleneck on {M}-bit blocks "
                "and would destroy per-slot information. Use h_group >= group_bits."
            )
        if self.group_cond not in ("none", "bias", "film"):
            raise ValueError(f"group_cond must be none|bias|film, got {self.group_cond!r}")
        if self.group_depth not in (1, 2):
            raise ValueError(f"group_depth must be 1 or 2, got {self.group_depth}")
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(dd: dict) -> "MLPConfig":
        dd = dict(dd)
        dd["fourier"] = tuple(dd["fourier"])
        known = {f.name for f in fields(MLPConfig)}
        unknown = set(dd) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return MLPConfig(**dd)  # missing keys fall back to defaults


# ---------------------------------------------------------------------------
# convenience constructors
# ---------------------------------------------------------------------------

def flat_config(d: int, **kw) -> MLPConfig:
    return MLPConfig(d=d, **kw).validate()


def grouped_config(n_groups: int, group_bits: int, reward: bool = True,
                   **kw) -> MLPConfig:
    """K groups of M bits, plus one reward coordinate if reward=True.

    GPT-2, 10 tokens of 15 bits:  grouped_config(10, 15)          -> d = 151
    4x4 Sudoku, 4 rows of 8 bits: grouped_config(4, 8)            -> d =  33
    4x4 Sudoku, 16 cells of 2:    grouped_config(16, 2, h_group=8)-> d =  33
    """
    d = n_groups * group_bits + (1 if reward else 0)
    return MLPConfig(d=d, n_groups=n_groups, group_bits=group_bits, **kw).validate()


class LevelEmbedding(nn.Module):
    """Features of the noise level.  Built around beta, not a, because beta is
    the natural parameter scale and a compresses everything into [0.9, 1]."""

    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.register_buffer("freqs", torch.tensor(cfg.fourier))
        self.level_on = cfg.level_on

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if a.dim() == 1:
            a = a.unsqueeze(1)
        if a.dim() != 2 or a.shape[1] != 1:
            raise ValueError(f"expected a of shape (B,) or (B,1), got {tuple(a.shape)}")
        lb = torch.log1p(beta_of_a(a))
        carrier = a if self.level_on == "a" else lb
        ang = 2 * math.pi * carrier * self.freqs
        return torch.cat([a, lb, torch.sin(ang), torch.cos(ang)], 1)


class GroupEncoder(nn.Module):
    """One shared encoder applied to each of the K contiguous M-bit blocks.

    Slot identity is restored by per-slot conditioning:
      none  h = W t
      bias  h = W t + b_k
      film  h = (1 + g_k) * W t + b_k
    Both g and b are zero-initialised, so film starts exactly at plain shared
    linear and only departs from it if the data pays for it.
    """

    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg
        K, M, H = cfg.n_groups, cfg.group_bits, cfg.h_group
        if cfg.group_depth == 1:
            self.proj = nn.Linear(M, H)
        else:
            self.proj = nn.Sequential(nn.Linear(M, H), nn.SiLU(), nn.Linear(H, H))
        if cfg.group_cond in ("bias", "film"):
            self.b = nn.Parameter(torch.zeros(K, H))
        if cfg.group_cond == "film":
            self.g = nn.Parameter(torch.zeros(K, H))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, K, M = u.shape[0], cfg.n_groups, cfg.group_bits
        blocks = u[:, : K * M].view(B, K, M)      # pure reshape, no arithmetic
        h = self.proj(blocks)                     # (B, K, H) via the SAME weights
        if cfg.group_cond == "film":
            h = (1.0 + self.g) * h + self.b
        elif cfg.group_cond == "bias":
            h = h + self.b
        return torch.cat([h.reshape(B, K * cfg.h_group), u[:, K * M:]], dim=1)


class ResBlock(nn.Module):
    """Pre-activation residual block with FiLM conditioning on the level."""

    def __init__(self, cfg: MLPConfig):
        super().__init__()
        w = cfg.width
        self.norm = nn.LayerNorm(w)
        self.film = nn.Linear(cfg.n_level, 2 * w)
        self.l1 = nn.Linear(w, w)
        self.l2 = nn.Linear(w, w)
        for m in (self.l2, self.film):  # start as the identity
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, h, e):
        g, b = self.film(e).chunk(2, dim=1)
        r = (1.0 + g) * self.norm(h) + b
        return h + self.l2(F.silu(self.l1(F.silu(r))))


class TiltMLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.level = LevelEmbedding(cfg)
        self.encoder = GroupEncoder(cfg) if cfg.grouped else None
        self.lin_in = nn.Linear(cfg.feat_dim + cfg.n_level, cfg.width)
        self.blocks = nn.ModuleList([ResBlock(cfg) for _ in range(cfg.n_blocks)])
        self.norm_out = nn.LayerNorm(cfg.width)
        self.lin_out = nn.Linear(cfg.width, cfg.d)
        self.gain = nn.Sequential(
            nn.Linear(cfg.n_level, cfg.gain_hidden), nn.SiLU(),
            nn.Linear(cfg.gain_hidden, cfg.d))
        if cfg.skip:
            nn.init.zeros_(self.lin_out.weight)
            nn.init.zeros_(self.lin_out.bias)

    def forward(self, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if a.dim() == 1:
            a = a.unsqueeze(1)
        if u.shape[-1] != self.cfg.d:
            raise ValueError(f"expected u of width {self.cfg.d}, got {u.shape[-1]}")
        e = self.level(a)
        x = u / math.sqrt(self.cfg.d)
        if self.encoder is not None:
            x = self.encoder(x)
        h = self.lin_in(torch.cat([x, e], dim=1))
        for blk in self.blocks:
            h = blk(h, e)
        r = self.lin_out(self.norm_out(h))
        if self.cfg.skip:
            # raw u, not the scaled/grouped features: the mean-field skip is
            # identical in the flat and grouped variants.
            return torch.tanh(beta_of_a(a) * u + self.gain(e) * r)
        return torch.tanh(r)

    def natural(self, h: torch.Tensor) -> torch.Tensor:
        """Evaluate at a natural parameter h, via the normalised parameterisation.

        This is the entry point the sampler uses, and the one that respects the
        invariance G(v, a) = G(cv, a') exactly.
        """
        from .noising import normalize_natural

        v, a = normalize_natural(h, self.cfg.d)
        return self.forward(v, a)

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
        net = TiltMLP(cfg)
        net.load_state_dict(ck["state_dict"])
        net.eval()
        return net, ck.get("extra", {})


if __name__ == "__main__":  # python -m tilt.models
    torch.manual_seed(0)
    cases = {
        "gpt2 flat":       flat_config(151, width=512, n_blocks=6),
        "gpt2 grouped":    grouped_config(10, 15, width=512, n_blocks=6),
        "gpt2 grp nocond": grouped_config(10, 15, width=512, n_blocks=6,
                                          group_cond="none"),
        "sudoku flat":     flat_config(33, width=256, n_blocks=4),
        "sudoku by row":   grouped_config(4, 8, width=256, n_blocks=4),
        "sudoku by cell":  grouped_config(16, 2, width=256, n_blocks=4, h_group=8),
    }
    for name, cfg in cases.items():
        net = TiltMLP(cfg)
        u = torch.randn(7, cfg.d)
        a = torch.rand(7)
        y = net(u, a)
        assert y.shape == (7, cfg.d), y.shape
        assert y.abs().max() <= 1.0
        # zero-init lin_out => the net starts exactly at mean field
        assert torch.allclose(y, torch.tanh(beta_of_a(a[:, None]) * u), atol=1e-6)
        print(f"{name:16s} {cfg.tag:44s} {net.n_params():>10,} params")
