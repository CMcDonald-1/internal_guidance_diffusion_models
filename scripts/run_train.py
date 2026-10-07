"""Train: python scripts/run_train.py --data data/run --out ckpt/net.pt --steps 3000"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from tilt.data import load_dataset
from tilt.evaluate import sse_eval_fn
from tilt.models import MLPConfig
from tilt.train import train

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/run")
p.add_argument("--out", default="ckpt/net.pt")
p.add_argument("--width", type=int, default=96)
p.add_argument("--blocks", type=int, default=4)
p.add_argument("--no-skip", action="store_true")
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--batch-size", type=int, default=1024)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--dt-weight", action="store_true")
p.add_argument("--lambda-max", type=float, default=1.0)
p.add_argument("--mode1-prob", type=float, default=0.0,
               help="probability per training row of drawing the infill-focusing "
                    "profile (sparse random-k-subset q) instead of the reward-"
                    "focusing profile (q=ones); 0 (default) never draws it, "
                    "reproducing lambda-only runs exactly (write-up Section 8.4).")
p.add_argument("--lambda1-prob", type=float, default=0.25,
               help="probability per row (when lambda-max > 1 or mode1-prob > 0) "
                    "of forcing lambda=1 (no reward-axis stretch) instead of "
                    "drawing it log-uniform on [1, lambda-max].")
p.add_argument("--beta-focus-lo", type=float, default=None,
               help="lower end of a 'hard' beta sub-range to oversample "
                    "(requires --beta-focus-hi and --beta-focus-prob > 0); "
                    "does not narrow the dataset's own beta_lo/beta_hi range.")
p.add_argument("--beta-focus-hi", type=float, default=None,
               help="upper end of the 'hard' beta sub-range, see --beta-focus-lo.")
p.add_argument("--beta-focus-prob", type=float, default=0.0,
               help="probability per row of drawing beta from [beta-focus-lo, "
                    "beta-focus-hi] instead of the dataset's full [beta_lo, beta_hi] "
                    "range; 0 (default) never does this, reproducing existing runs "
                    "exactly.")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default=None,
               help="cuda / cpu / cuda:0 etc.; default auto-picks cuda if available")
p.add_argument("--log-every", type=int, default=500)
p.add_argument("--eval-every", type=int, default=None)
p.add_argument("--wandb", action="store_true", help="log train/eval metrics to Weights & Biases")
p.add_argument("--wandb-project", default="tilted-diffusions")
p.add_argument("--wandb-entity", default=None,
               help="W&B team/entity to log under (a single account can belong to multiple teams)")
p.add_argument("--wandb-name", default=None, help="optional run name; auto-generated if omitted")
p.add_argument("--wandb-tags", default=None,
               help="comma-separated tags for the W&B run, e.g. "
                    "'sudoku,reward,lambda3.5,w1024b8'")
p.add_argument("--wandb-notes", default=None,
               help="free-text note for the W&B run describing why it exists "
                    "(what question it's trying to answer)")
p.add_argument("--n-eval-betas", type=int, default=6,
               help="log-spaced beta grid size for the test-SSE-vs-exact-operator eval")
p.add_argument("--eval-n", type=int, default=2048, help="samples per beta for eval")
p.add_argument("--eval-beta-hi", type=float, default=20.0,
               help="cap on the eval beta grid (below the tanh-saturation point, "
                    "where beta(a)*u swamps float32 precision and both net and exact "
                    "operator collapse to +-1 regardless of training). Training itself "
                    "still uses the dataset's full beta_lo-beta_hi range.")
p.add_argument("--no-eval", action="store_true", help="skip the exact-operator eval entirely")
p.add_argument("--jac-random-n", type=int, default=0,
               help="if > 0, also run the randomized-pair Jacobian symmetry check "
                    "(jacobian_symmetry_random) at each eval beta with this many "
                    "points per beta, logged as test/jac_symmetry_random/<beta>/*. "
                    "0 (default) disables it.")
p.add_argument("--jac-diag-n", type=int, default=0,
               help="if > 0, also run the Jacobian diagonal self-consistency check "
                    "(jacobian_diagonal) at each eval beta with this many points "
                    "per beta, logged as test/jac_diagonal/<beta>/*. "
                    "0 (default) disables it.")
a = p.parse_args()

_, _, target, meta = load_dataset(a.data)
cfg = MLPConfig(d=meta["d"], width=a.width, n_blocks=a.blocks, skip=not a.no_skip)

eval_fn = None
if not a.no_eval:
    eval_beta_hi = min(a.eval_beta_hi, meta["beta_hi"])
    eval_betas = np.logspace(
        np.log10(meta["beta_lo"]), np.log10(eval_beta_hi), a.n_eval_betas
    ).tolist()
    eval_fn = sse_eval_fn(target, eval_betas, n=a.eval_n,
                          jac_random_n=a.jac_random_n, jac_diag_n=a.jac_diag_n)

wandb_tags = [t.strip() for t in a.wandb_tags.split(",") if t.strip()] if a.wandb_tags else None

beta_focus_range = None
if a.beta_focus_lo is not None and a.beta_focus_hi is not None:
    beta_focus_range = (a.beta_focus_lo, a.beta_focus_hi)

train(a.data, a.out, cfg, steps=a.steps, batch_size=a.batch_size, lr=a.lr,
      dt_weight=a.dt_weight, lambda_max=a.lambda_max, mode1_prob=a.mode1_prob,
      lambda1_prob=a.lambda1_prob, beta_focus_range=beta_focus_range,
      beta_focus_prob=a.beta_focus_prob, seed=a.seed,
      log_every=a.log_every, eval_every=a.eval_every, eval_fn=eval_fn,
      use_wandb=a.wandb, wandb_project=a.wandb_project, wandb_run_name=a.wandb_name,
      wandb_entity=a.wandb_entity, wandb_tags=wandb_tags, wandb_notes=a.wandb_notes,
      device=a.device)
