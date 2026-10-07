"""Sudoku 6x6 check 3: memorization -- do reward-tilted samples exactly match a

training-set row?

Checkpoint : ckpt/sudoku6_v2_l10_xf_sc_s8264_180k.pt
             (TransformerTilt, self-conditioned, d=109)
Dataset    : data/sudoku6_v2  (grid-backed .npy, n=5,000,000 rows, row_noise=0.02,
             p_inject=0.05, 249,903 valid/248,778 distinct-valid per meta.json)

Draws n=1000 samples at tau=tau_98 (the tau where the EXACT closed-form success
rate first reaches 98%, same closed-form anchor used by
eval/sudoku6_check2_reward_logodds.py's tau grid -- see build_tau_grid), decodes
each to a (6,6) grid, and checks EXACT membership against every one of the
5,000,000 raw training rows (not just the subset that happen to be valid
solutions -- "does this generated grid appear anywhere in the training set at
all" is the literal question asked, and `GridBackedX.grids` is the full
population, cheap to hold as a hash set: 5M x 36 bytes = 180MB).

This is deliberately narrower than the more elaborate `scripts/
run_sudoku6_memorization.py` (which additionally compares against a uniform-
valid-grid null and a held-out-training ceiling via nearest-neighbor Hamming
distance) -- that script is UNVERIFIED reference material per eval/README.md,
and this check only does what was actually asked: an exact-match count at one
tau, nothing more. A nearest-neighbor/null-comparison follow-up can be added
later as its own check if wanted.

Usage:
    python eval/sudoku6_check3_memorization.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.checkpoints import load_premier  # noqa: E402
from common.reward_check import logit  # noqa: E402
from common.sample_config import K_DEFAULTS, SCHEDULE, T  # noqa: E402

from tilt.data_grids import GridBackedX  # noqa: E402
from tilt.sudoku6 import decode_grid_batch, is_valid_sudoku6_batch, success_rate_exact  # noqa: E402

CKPT = ROOT / "ckpt" / "sudoku6_v2_l10_xf_sc_s8264_180k.pt"
DATA = ROOT / "data" / "sudoku6_v2"
ROW_NOISE = 0.02
P_INJECT = 0.05


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--row-noise", type=float, default=ROW_NOISE)
    p.add_argument("--p-inject", type=float, default=P_INJECT)
    p.add_argument("--target-p", type=float, default=0.98,
                   help="exact success rate defining tau_98 (see build_tau_grid)")
    p.add_argument("--n", type=int, default=1000, help="samples to draw at tau_98")
    p.add_argument("--K", type=int, default=None, help="override K_DEFAULTS['sudoku6']")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "sudoku6"))
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    K = args.K if args.K is not None else K_DEFAULTS["sudoku6"]

    shim, info, sl, decode, needs_net, drift_arg = load_premier(args.ckpt, device)
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})  device={device}")
    if not info["selfcond"]:
        raise SystemExit("this checkpoint is not self-conditioned -- expected the "
                          "premier sudoku6 transformer/self-cond checkpoint")

    d = shim.net.cfg.d
    p0 = success_rate_exact(args.row_noise, args.p_inject, tau=0.0)
    tau_98 = (logit(args.target_p) - logit(p0)) / 2.0
    exact_at_tau98 = success_rate_exact(args.row_noise, args.p_inject, tau=tau_98)
    print(f"d={d}  p0={p0:.4%}  tau_98={tau_98:.4f}  exact success rate there={exact_at_tau98:.4%}")
    print(f"sampling config: T={T}, K={K}, schedule={SCHEDULE!r}, n={args.n}")

    h0 = torch.zeros(d)
    h0[-1] = tau_98
    raw = sl(drift_arg, d=d, n=args.n, h=h0, T=T, K=K, schedule=SCHEDULE, seed=args.seed)
    hard = decode(raw, drift_arg, h0)

    n_reward_hit = int((hard[:, -1] > 0).sum())
    grids = decode_grid_batch(hard[:, :108])
    valid_mask = is_valid_sudoku6_batch(grids).cpu().numpy().astype(bool)
    n_valid = int(valid_mask.sum())
    print(f"generated: {args.n} samples -- reward bit +1: {n_reward_hit} "
          f"({100*n_reward_hit/args.n:.2f}%), actually valid grids: {n_valid} "
          f"({100*n_valid/args.n:.2f}%)")

    gen_codes = grids.cpu().numpy().reshape(args.n, 36).astype(np.uint8)
    n_distinct = len(np.unique(gen_codes, axis=0))
    print(f"distinct generated grids: {n_distinct}/{args.n}")

    print(f"\nloading training set from {args.data} ...")
    x = GridBackedX.load(args.data)
    print(f"  {x.n:,} rows, {x.cells} cells/row")
    train_set = {row.tobytes() for row in x.grids}
    print(f"  {len(train_set):,} distinct training rows")

    matches = np.fromiter((row.tobytes() in train_set for row in gen_codes), dtype=bool, count=args.n)
    n_match = int(matches.sum())
    n_match_valid = int((matches & valid_mask).sum())

    print(f"\n=== exact-match results (n={args.n} samples at tau_98={tau_98:.4f}) ===")
    print(f"  matched a training-set row: {n_match}/{args.n} ({100*n_match/args.n:.3f}%)")
    print(f"  of those, also a genuinely valid grid: {n_match_valid}/{n_valid if n_valid else 1} "
          f"valid samples ({100*n_match_valid/max(n_valid,1):.3f}%)")
    if n_match:
        idx = np.flatnonzero(matches)[:5]
        print(f"  first matching sample indices: {idx.tolist()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    summary = pd.DataFrame([{
        "tau_98": tau_98, "target_p": args.target_p, "exact_success_rate": exact_at_tau98,
        "n": args.n, "n_reward_hit": n_reward_hit, "n_valid": n_valid,
        "n_distinct_generated": n_distinct, "n_train_rows": x.n,
        "n_distinct_train_rows": len(train_set),
        "n_exact_match": n_match, "exact_match_rate": n_match / args.n,
        "n_exact_match_and_valid": n_match_valid,
        "K": K, "T": T, "schedule": SCHEDULE, "seed": args.seed,
    }])
    summary.to_csv(out_dir / "check3_memorization.csv", index=False)
    print(f"\nsaved: {out_dir / 'check3_memorization.csv'}")


if __name__ == "__main__":
    main()
