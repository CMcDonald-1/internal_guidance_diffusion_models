"""Sudoku 4x4 lollipop: true G vs. net G at a=0.5, tau in {0, 4}.

Checkpoint : ckpt/sudoku_reward_v2_infill_mode1p25_lambda5_w1024b8_30k.pt
Dataset    : data/sudoku_reward_v2  (d=33)

a=0.5 sits in the hard mid-signal region flagged by
eval/sudoku4x4_check1_denoiser_ood.py (SSE peaks around a~0.5-0.6 for every
tau tested there). This script visualizes WHERE across the 33 coordinates
the net's denoiser prediction diverges from the exact ground truth at that
same a, at two tilt strengths (tau=0, untilted; tau=4, near this
checkpoint's own lambda_max=5.0 training ceiling).

Uses tilt/evaluate.py's private _typical_inputs/_ground_truth/_query helpers
directly (same pattern as the project's earlier ad hoc lollipop comparisons)
rather than
duplicating that logic, plus the existing plot_lollipop helper.

Usage:
    python eval/sudoku4x4_lollipop_a0p5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tilt.data import load_dataset  # noqa: E402
from tilt.evaluate import _ground_truth, _query, _typical_inputs, plot_lollipop  # noqa: E402
from tilt.load_ckpt import load_tilt_checkpoint  # noqa: E402
from tilt.noising import beta_of_a  # noqa: E402

CKPT = ROOT / "ckpt" / "sudoku_reward_v2_infill_mode1p25_lambda5_w1024b8_30k.pt"
DATA = ROOT / "data" / "sudoku_reward_v2"
OUT_DIR = ROOT / "eval" / "results" / "sudoku4x4"

A_LEVEL = 0.5
TAUS = [0.0, 4.0]
SEED = 0
N_SAMPLES = 5


def main() -> None:
    _x_train, _frozen, target, _meta = load_dataset(str(DATA))
    if target is None or not hasattr(target, "tilt_direction"):
        raise SystemExit("need a RewardAugmented target with tilt_direction()")

    net, info = load_tilt_checkpoint(str(CKPT), map_location="cpu")
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})")

    beta = float(beta_of_a(torch.tensor(A_LEVEL)))
    print(f"a={A_LEVEL}  beta={beta:.4f}")

    for tau in TAUS:
        mode = "ood" if tau != 0.0 else "in_dist"
        with torch.no_grad():
            _x, u, a, h_extra = _typical_inputs(
                target, beta, n=N_SAMPLES, seed=SEED, mode=mode, tau=tau)
            gt = _ground_truth(target, u, a, h_extra)
            pred = _query(net, u, a, h_extra, d=target.d)

        per_sample_sse = (pred - gt).pow(2).sum(dim=1)
        print(f"tau={tau}: per-sample sse = {[f'{v:.4f}' for v in per_sample_sse.tolist()]}")

        fig, axes = plot_lollipop(
            gt, pred,
            title=f"Sudoku 4x4 d={target.d}  a={A_LEVEL} (beta={beta:.3f})  tau={tau}  "
                  f"({N_SAMPLES} samples)",
        )
        # move the legend outside the axes -- default in-axes placement can
        # obscure the final few coordinates.
        axes[0].get_legend().remove()
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"lollipop_a0p5_tau{tau:g}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
