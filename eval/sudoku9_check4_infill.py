"""Sudoku 9x9 check 4: infill check.

Generates P uniform-random valid 9x9 solutions (`tilt.sudoku9_fast.
sample_valid_solutions_fast`), for each of several `n_removed` values masks that
many of the 81 cells (a fresh random subset per puzzle), builds a combined
reward-tilt + cell-fix natural parameter (the same construction as
`tilt/sudoku9_infill.py::build_infill_tilt`/`infill_success_rate`, inlined here
in a batched, per-row form since every puzzle's mask differs), and samples the
premier checkpoint. Reports, per `n_removed`:

  - p_valid         : decoded grid is a genuinely valid 9x9 solution.
  - p_honors_given  : every REVEALED (non-removed) cell's 4 bits match the
                       original solution's sign.
  - p_success       : both at once -- the criterion that actually matters for
                       infilling (NOT exact match on the removed cells, since a
                       lightly-constrained puzzle can have more than one valid
                       completion).

self_cond_mode="carry" is used by default (not "two_pass"), per this
checkpoint's already-established finding (see eval/sudoku9_selfcond_mode_check.py,
eval/README.md) that carry gives substantially higher actual grid validity on
THIS premier checkpoint specifically -- a checkpoint-specific empirical choice,
not a general recommendation. tau/strength default to 3.0/3.0, matching the
"keep strength comparable to tau" guidance carried over from the 4x4 infill
investigation (tilt/sudoku_infill.py) and the tau=3 point already used
elsewhere in this checkpoint's eval battery.

Usage:
    python eval/sudoku9_check4_infill.py
    python eval/sudoku9_check4_infill.py --n-removed 10 40 70 --n-puzzles 20 --n-per-puzzle 5
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # standalone subprocess run: no GUI backend available
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.checkpoints import load_premier  # noqa: E402
from common.reward_check import clopper_pearson_ci  # noqa: E402
from common.sample_config import K_DEFAULTS, SCHEDULE, T  # noqa: E402

from tilt.sudoku9 import D, is_valid_sudoku9_batch  # noqa: E402
from tilt.sudoku9_fast import sample_valid_solutions_fast  # noqa: E402

CKPT = ROOT / "ckpt" / "sudoku9_v1_l17_d512x8_xf_sc_s8264_180k.pt"
N_CELLS = 81
BITS_PER_CELL = 4


def build_batch_infill_tilt(sols: torch.Tensor, given_mask: torch.Tensor, d: int,
                             tau: float, strength: float) -> torch.Tensor:
    """Per-row combined reward + cell-fix natural parameter.

    sols:       (P, D) {-1,1} core solution bits.
    given_mask: (P, 81) bool, True where that cell is REVEALED (not removed).

    Returns h: (P, d) with `tau` on the reward coordinate (index -1) and
    `strength * sign(sol)` on every revealed cell's 4 bits, 0 elsewhere
    (including every removed cell and, implicitly, the padding beyond D if
    d > D + 1). Direct batched analogue of `tilt/sudoku9_infill.py::
    build_infill_tilt`, which only handles one puzzle/mask at a time.
    """
    p = sols.shape[0]
    h = torch.zeros(p, d)
    h[:, -1] = tau
    cell_bits = given_mask.repeat_interleave(BITS_PER_CELL, dim=1)  # (P, D) bool
    h[:, :D] = torch.where(cell_bits, strength * torch.sign(sols), torch.zeros_like(sols))
    return h


def random_given_masks(n_puzzles: int, n_removed: int, generator: torch.Generator) -> torch.Tensor:
    """(n_puzzles, 81) bool, True = revealed. A fresh random `n_removed`-cell
    subset is masked out independently for each puzzle."""
    mask = torch.ones(n_puzzles, N_CELLS, dtype=torch.bool)
    for i in range(n_puzzles):
        remove_idx = torch.randperm(N_CELLS, generator=generator)[:n_removed]
        mask[i, remove_idx] = False
    return mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--self-cond-mode", default="carry", choices=["two_pass", "carry"],
                   help="default carry: see module docstring for why, on this checkpoint")
    p.add_argument("--n-removed", type=int, nargs="+", default=[10, 20, 30, 40, 50, 60, 70],
                   help="number of the 81 cells to mask/remove, swept")
    p.add_argument("--n-puzzles", type=int, default=100)
    p.add_argument("--n-per-puzzle", type=int, default=20, help="SDE draws per puzzle")
    p.add_argument("--tau", type=float, default=3.0)
    p.add_argument("--strength", type=float, default=3.0)
    p.add_argument("--K", type=int, default=None, help="override K_DEFAULTS['sudoku9']")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "sudoku9"))
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    K_default = K_DEFAULTS["sudoku9"]
    K = args.K if args.K is not None else K_default
    suffix = f"_{args.self_cond_mode}" + ("" if K == K_default else f"_K{K}")

    shim, info, sl, decode, needs_net, drift_arg = load_premier(args.ckpt, device)
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})  "
          f"self_cond_mode={args.self_cond_mode!r}  device={device}")
    if not info["selfcond"]:
        raise SystemExit("this checkpoint is not self-conditioned -- expected the "
                          "premier sudoku9 transformer/self-cond checkpoint")

    d = shim.net.cfg.d
    assert d == D + 1, f"expected d={D + 1} (324 core + 1 reward), got {d}"
    print(f"d={d}  n_removed={args.n_removed}  n_puzzles={args.n_puzzles}  "
          f"n_per_puzzle={args.n_per_puzzle}  tau={args.tau}  strength={args.strength}  "
          f"T={T}, K={K}, schedule={SCHEDULE!r}")

    mask_gen = torch.Generator().manual_seed(args.seed)
    sol_gen = torch.Generator().manual_seed(args.seed + 1)
    sols = sample_valid_solutions_fast(args.n_puzzles, generator=sol_gen)  # (P, D)
    print(f"generated {args.n_puzzles} uniform-random valid 9x9 solutions")

    rows = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        for i, n_removed in enumerate(args.n_removed, start=1):
            given_mask = random_given_masks(args.n_puzzles, n_removed, mask_gen)  # (P, 81)

            h_per_puzzle = build_batch_infill_tilt(sols, given_mask, d, args.tau, args.strength)
            h = h_per_puzzle.repeat_interleave(args.n_per_puzzle, dim=0)
            sol_rep = sols.repeat_interleave(args.n_per_puzzle, dim=0)
            given_bits_rep = (given_mask.repeat_interleave(BITS_PER_CELL, dim=1)
                              .repeat_interleave(args.n_per_puzzle, dim=0))

            n_total = args.n_puzzles * args.n_per_puzzle
            raw = sl(drift_arg, d=d, n=n_total, h=h, T=T, K=K, schedule=SCHEDULE,
                     seed=args.seed, self_cond_mode=args.self_cond_mode)
            hard = decode(raw, drift_arg, h)

            core = hard[:, :D]
            valid = is_valid_sudoku9_batch(core)
            bit_ok = torch.where(given_bits_rep, torch.sign(core) == torch.sign(sol_rep),
                                  torch.ones_like(given_bits_rep))
            honors = bit_ok.all(dim=1)
            success = valid & honors

            n_valid, n_honors, n_success = int(valid.sum()), int(honors.sum()), int(success.sum())
            p_valid, p_honors, p_success = (n_valid / n_total, n_honors / n_total, n_success / n_total)
            v_lo, v_hi = clopper_pearson_ci(n_valid, n_total)
            hn_lo, hn_hi = clopper_pearson_ci(n_honors, n_total)
            s_lo, s_hi = clopper_pearson_ci(n_success, n_total)

            row = {
                "n_removed": n_removed, "n_given": N_CELLS - n_removed,
                "tau": args.tau, "strength": args.strength, "K": K,
                "self_cond_mode": args.self_cond_mode,
                "n_puzzles": args.n_puzzles, "n_per_puzzle": args.n_per_puzzle, "n_total": n_total,
                "p_valid": p_valid, "p_valid_ci_lo": v_lo, "p_valid_ci_hi": v_hi,
                "p_honors_given": p_honors, "p_honors_ci_lo": hn_lo, "p_honors_ci_hi": hn_hi,
                "p_success": p_success, "p_success_ci_lo": s_lo, "p_success_ci_hi": s_hi,
            }
            rows.append(row)
            print(f"  [{i}/{len(args.n_removed)}] n_removed={n_removed:2d} "
                  f"(given={N_CELLS - n_removed:2d})  "
                  f"p_valid={p_valid:.4f} [{v_lo:.4f},{v_hi:.4f}]  "
                  f"p_honors_given={p_honors:.4f} [{hn_lo:.4f},{hn_hi:.4f}]  "
                  f"p_success={p_success:.4f} [{s_lo:.4f},{s_hi:.4f}]")

            pd.DataFrame(rows).to_csv(out_dir / f"check4_infill{suffix}.csv", index=False)

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    specs = [
        ("p_valid", "p_valid_ci_lo", "p_valid_ci_hi", "tab:green", "P(valid grid)"),
        ("p_honors_given", "p_honors_ci_lo", "p_honors_ci_hi", "tab:orange", "P(honors given cells)"),
        ("p_success", "p_success_ci_lo", "p_success_ci_hi", "tab:blue", "P(valid AND honors given)"),
    ]
    for col, lo_col, hi_col, color, label in specs:
        yerr = np.vstack([df[col] - df[lo_col], df[hi_col] - df[col]])
        ax.errorbar(df["n_given"], df[col], yerr=yerr, fmt="o-", capsize=3,
                    color=color, label=label)
    ax.set_xlabel("number of cells given (81 - n_removed)")
    ax.set_ylabel("probability")
    ax.set_ylim(-0.02, 1.02)
    ax.invert_xaxis()  # harder (fewer given cells) on the right
    ax.set_title(f"Sudoku 9x9 infill check  (tau={args.tau}, strength={args.strength}, "
                 f"K={K}, self_cond_mode={args.self_cond_mode!r},\n"
                 f"{args.n_puzzles} puzzles x {args.n_per_puzzle} draws/puzzle)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"check4_infill{suffix}.png", dpi=150)

    print(f"\nsaved: {out_dir / f'check4_infill{suffix}.csv'}")
    print(f"saved: {out_dir / f'check4_infill{suffix}.png'}")


if __name__ == "__main__":
    main()
