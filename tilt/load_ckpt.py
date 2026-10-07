"""One place that knows how to load any tilt checkpoint.

Every evaluation script needs the same three facts about a checkpoint: which
class to build, whether it is self-conditioned (the sampler entry point
differs), and a human label. Before the transformer arm existed, all three
were read off the FILENAME:

    _NAME_RE = r"_(flat|matrix)_(sc|nosc)_s(\\w+?)_(\\d+k)\\.pt$"

which silently fails on `sudoku6_v2_l10_xf_sc_s8264_180k.pt` -- the regex does
not match, `selfcond` falls back to False, and a TransformerTilt checkpoint
gets handed to TiltMLP.load. That produces a state-dict error at best and a
wrong-model evaluation at worst.

TransformerTilt.save writes `"arch": "transformer"` into the checkpoint, so
the architecture is available from the file itself. This module prefers that
and uses the filename only for what genuinely is not stored (the MLP arms'
self-conditioning flag, and the seed).
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

# `xf` added; seed group loosened so v2 names parse too.
NAME_RE = re.compile(r"_(flat|matrix|xf)_(sc|nosc)_s(\w+?)_(\d+k)\.pt$")


def parse_name(path: str | Path) -> dict:
    """Best-effort (arch, selfcond, seed, steps) from the filename."""
    m = NAME_RE.search(Path(path).name)
    if not m:
        return {"arch": "unknown", "selfcond": False,
                "seed": "unknown", "steps_tag": "unknown"}
    arch, sc, seed, steps = m.groups()
    return {"arch": arch, "selfcond": sc == "sc", "seed": seed, "steps_tag": steps}


def load_tilt_checkpoint(path: str | Path, map_location="cpu"):
    """Return (net, info) with the right class for this checkpoint.

    info: {"arch", "selfcond", "seed", "steps_tag", "kind"} where `kind` is
    "transformer" | "selfcond_mlp" | "mlp".
    """
    path = Path(path)
    info = parse_name(path)

    ck = torch.load(path, map_location=map_location, weights_only=False)
    arch = ck.get("arch") if isinstance(ck, dict) else None

    if arch == "transformer":
        from .models_transformer import TransformerTilt

        net, _extra = TransformerTilt.load(path, map_location=map_location)
        info["arch"] = "xf"
        # authoritative: the saved config, not the filename
        info["selfcond"] = bool(ck.get("cfg", {}).get("self_cond", info["selfcond"]))
        info["kind"] = "transformer"
    elif info["selfcond"]:
        from .models_selfcond import SelfCondTiltMLP

        net, _extra = SelfCondTiltMLP.load(path, map_location=map_location)
        info["kind"] = "selfcond_mlp"
    else:
        from .models import TiltMLP

        net, _extra = TiltMLP.load(path, map_location=map_location)
        info["kind"] = "mlp"

    net.eval()
    return net, info


def sample_fns(info: dict):
    """(sl_sample_fn, decode_fn, needs_net_object) for this checkpoint kind.

    The self-conditioned samplers take the NET; the plain ones take
    `net.natural`. TransformerTilt exposes both `natural` and
    `natural_self_cond`, so it routes by its own self_cond flag.
    """
    from .sampler import decode_drift, sl_sample
    from .sampler_selfcond import decode_drift_selfcond, sl_sample_selfcond

    if info["selfcond"]:
        return sl_sample_selfcond, decode_drift_selfcond, True
    return sl_sample, decode_drift, False
