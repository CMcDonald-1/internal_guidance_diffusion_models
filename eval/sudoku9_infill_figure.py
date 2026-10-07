"""Sudoku 9x9 infill: one figure showing a partially-filled puzzle and its
generated completion, via `tilt/sudoku9_viz.py::plot_infill_sample`.

Draws ONE random valid solution, removes `--n-removed` of its 81 cells, builds
the combined reward + cell-fix tilt (single-puzzle analogue of
`eval/sudoku9_check4_infill.py::build_batch_infill_tilt`), and samples the
premier checkpoint. Retries with a fresh batch (new seed) until at least one
draw is a genuine SUCCESS (valid grid AND every given cell honored), or
`--max-attempts` is exhausted -- `eval/sudoku9_check4_infill.py`'s sweep found
~85-90% per-draw success at n_removed=50/60 (n_removed=51 sits between them),
so in practice this should essentially never need more than one attempt.

Usage:
    python eval/sudoku9_infill_figure.py
    python eval/sudoku9_infill_figure.py --n-removed 51 --n-per-attempt 30 --max-attempts 5
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.checkpoints import load_premier  # noqa: E402
from common.sample_config import K_DEFAULTS, SCHEDULE, T  # noqa: E402

from tilt.sudoku9 import D, is_valid_sudoku9_batch  # noqa: E402
from tilt.sudoku9_fast import sample_valid_solutions_fast  # noqa: E402
from tilt.sudoku9_viz import plot_infill_sample  # noqa: E402

CKPT = ROOT / "ckpt" / "sudoku9_v1_l17_d512x8_xf_sc_s8264_180k.pt"
N_CELLS = 81
BITS_PER_CELL = 4


def build_infill_tilt(sol: torch.Tensor, given_cells, d: int, tau: float,
                       strength: float) -> torch.Tensor:
    """Single-puzzle version of `eval/sudoku9_check4_infill.py`'s batched builder."""
    h = torch.zeros(d)
    h[-1] = tau
    for r, c in given_cells:
        b = BITS_PER_CELL * (9 * r + c)
        h[b:b + BITS_PER_CELL] = strength * torch.sign(sol[b:b + BITS_PER_CELL])
    return h


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--self-cond-mode", default="carry", choices=["two_pass", "carry"])
    p.add_argument("--n-removed", type=int, default=51)
    p.add_argument("--tau", type=float, default=3.0)
    p.add_argument("--strength", type=float, default=3.0)
    p.add_argument("--K", type=int, default=None, help="override K_DEFAULTS['sudoku9']")
    p.add_argument("--n-per-attempt", type=int, default=30)
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=str(ROOT / "eval" / "results" / "sudoku9" /
                                          "infill_figure_sample.png"))
    p.add_argument("--left-title", default=None,
                   help="override the left-panel title (default: auto, "
                        "'blank start' when --n-removed 81)")
    p.add_argument("--right-title", default=None,
                   help="override the right-panel title (default: auto, "
                        "'unconditional generation' when --n-removed 81)")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    K = args.K if args.K is not None else K_DEFAULTS["sudoku9"]

    shim, info, sl, decode, needs_net, drift_arg = load_premier(args.ckpt, device)
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})  "
          f"self_cond_mode={args.self_cond_mode!r}  device={device}")
    if not info["selfcond"]:
        raise SystemExit("this checkpoint is not self-conditioned -- expected the "
                          "premier sudoku9 transformer/self-cond checkpoint")

    d = shim.net.cfg.d
    assert d == D + 1, f"expected d={D + 1}, got {d}"

    sol_gen = torch.Generator().manual_seed(args.seed)
    mask_gen = torch.Generator().manual_seed(args.seed + 1)
    sol = sample_valid_solutions_fast(1, generator=sol_gen)[0]  # (D,)

    all_cells = [(r, c) for r in range(9) for c in range(9)]
    remove_idx = torch.randperm(N_CELLS, generator=mask_gen)[:args.n_removed].tolist()
    remove_set = {all_cells[i] for i in remove_idx}
    given_cells = [rc for rc in all_cells if rc not in remove_set]
    print(f"n_given={len(given_cells)}  n_removed={len(remove_set)}  tau={args.tau}  "
          f"strength={args.strength}  K={K}  T={T}, schedule={SCHEDULE!r}")

    h = build_infill_tilt(sol, given_cells, d, args.tau, args.strength)
    given_idx = torch.tensor(
        [BITS_PER_CELL * (9 * r + c) + j for r, c in given_cells for j in range(BITS_PER_CELL)],
        dtype=torch.long)

    found = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        for attempt in range(1, args.max_attempts + 1):
            raw = sl(drift_arg, d=d, n=args.n_per_attempt, h=h, T=T, K=K,
                     schedule=SCHEDULE, seed=args.seed + 1000 * attempt,
                     self_cond_mode=args.self_cond_mode)
            hard = decode(raw, drift_arg, h)
            core = hard[:, :D]
            valid = is_valid_sudoku9_batch(core)
            honors = (torch.sign(core[:, given_idx]) == torch.sign(sol[given_idx])).all(dim=1)
            success = valid & honors
            n_success = int(success.sum())
            print(f"  attempt {attempt}/{args.max_attempts}: "
                  f"{n_success}/{args.n_per_attempt} successes")
            if n_success > 0:
                found = hard[success][0]
                break

    if found is None:
        raise SystemExit(f"no success found in {args.max_attempts} attempts x "
                          f"{args.n_per_attempt} draws -- rerun with a higher "
                          f"--max-attempts or --n-per-attempt")

    if args.left_title is not None or args.right_title is not None:
        default_left = "blank start" if not given_cells else None
        default_right = "unconditional generation" if not given_cells else None
        titles = (args.left_title or default_left, args.right_title or default_right)
    elif not given_cells:
        titles = ("blank start", "unconditional generation")
    else:
        titles = None

    fig, axes = plot_infill_sample(given_cells, sol, found, path=args.out, titles=titles)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
