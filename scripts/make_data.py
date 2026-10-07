"""Generate a training set: python scripts/make_data.py --kind ising --d 16 -n 200000"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tilt.data import make_dataset
from tilt.targets import build_target

p = argparse.ArgumentParser()
p.add_argument("--kind", default="ising", choices=["product", "ising", "sudoku"])
p.add_argument("--d", type=int, default=16,
              help="ignored for --kind sudoku, which is fixed at d=32 (pass --d 32 explicitly)")
p.add_argument("--J0", type=float, default=1.5)
p.add_argument("--field", type=float, default=0.3)
p.add_argument("--reward", default=None,
              help="e.g. majority, parity, threshold:12 (product/ising only; "
                   "sudoku uses --p-inject instead)")
p.add_argument("--row-noise", type=float, default=0.0,
              help="sudoku only: per-row prob. of a fully random (possibly "
                   "invalid) row instead of a permutation (default: 0, no noise)")
p.add_argument("--p-inject", type=float, default=None,
              help="sudoku only: mixture weight on uniform_on_reward_set "
                   "(the 288 valid solutions) for reward injection")
p.add_argument("-n", type=int, default=200_000)
p.add_argument("--out", default="data/run")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--beta-lo", type=float, default=0.02)
p.add_argument("--beta-hi", type=float, default=50.0)
p.add_argument("--freeze-noise", action="store_true")
a = p.parse_args()

tgt = build_target(a.kind, a.d, seed=a.seed, J0=a.J0, field=a.field, reward=a.reward,
                   row_noise=a.row_noise, p_inject=a.p_inject)
path = make_dataset(tgt, a.n, a.out, seed=a.seed, freeze_noise=a.freeze_noise,
                    beta_lo=a.beta_lo, beta_hi=a.beta_hi)
print(f"wrote {path}  (d={tgt.d}, eff. support {tgt.effective_support():.0f}"
      + (f", P(r=+1)={tgt.success_rate():.4f}" if hasattr(tgt, "success_rate") else "") + ")")
