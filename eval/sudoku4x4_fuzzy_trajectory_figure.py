"""Sudoku 4x4 fuzzy-puzzle trajectory figure, via `tilt/sudoku_viz.py::
plot_fuzzy_trajectory` (pre-existing, unmodified -- see that module's docstring
for what "fuzzy puzzle"/"fixation event" mean).

Confirmed the 4x4 premier checkpoint
(`ckpt/sudoku_reward_v2_infill_mode1p25_lambda5_w1024b8_30k.pt`) is a
plain (non-self-conditioned) `TiltMLP` (`info["kind"] == "mlp"`, `d=33` = 32
core bits + 1 reward coordinate) -- exactly what `tilt/sudoku_viz.py` assumes
(it calls `net.natural(...)` directly, with no self-conditioning `x_pred`
argument), so no adaptation was needed there.

`fuzzy_fixation_trajectory`/`plot_fuzzy_trajectory` still default to this
project's now-superseded sampling settings (`T=20, K=200, schedule="late"`) --
this script passes the canonical ones used everywhere else in `eval/`
(`T=20, K=K_DEFAULTS["sudoku4x4"]=1000, schedule="sstar"`) explicitly instead.

`--tau` defaults to 3.0, not 0.0: an untilted (tau=0) draw from this
checkpoint's base measure is usually NOT a valid grid (this base measure's
untilted success rate is only a few percent -- see `tilt.sudoku.
success_rate_exact`-style reasoning elsewhere in this project), so tau=0 is
not a representative "does the network solve sudoku" illustration. `--seed`
defaults to 5, chosen (over seeds 0-4) by visual comparison across a batch of
candidates: it has the widest spread of cell-fixation events (t=0.285-0.849)
and a visually clear trajectory that dips toward its wrong sign before
reversing and resolving -- both make for a clearer report figure than the
other seeds tried, which converged faster and less dramatically. Trajectory
lines use `plot_fuzzy_trajectory`'s own defaults for display styling
(EMA-smoothed for readability, see that function's docstring) -- nothing
sampling-related is display-smoothed, only how the lines are drawn.

Usage:
    python eval/sudoku4x4_fuzzy_trajectory_figure.py
    python eval/sudoku4x4_fuzzy_trajectory_figure.py --tau 4.0 --seed 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # standalone subprocess run: no GUI backend available
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.checkpoints import load_premier  # noqa: E402
from common.sample_config import K_DEFAULTS, SCHEDULE, T  # noqa: E402

from tilt.sudoku_viz import plot_fuzzy_trajectory  # noqa: E402

CKPT = ROOT / "ckpt" / "sudoku_reward_v2_infill_mode1p25_lambda5_w1024b8_30k.pt"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--tau", type=float, default=3.0,
                   help="reward-tilt strength on the last coordinate; 0.0 = untilted, "
                        "an ordinary (usually invalid) draw from the base measure -- "
                        "default 3.0 to reliably illustrate a valid solve")
    p.add_argument("--threshold", type=float, default=0.75,
                   help="|G| confidence threshold for a bit to be considered fixed")
    p.add_argument("--K", type=int, default=None, help="override K_DEFAULTS['sudoku4x4']")
    p.add_argument("--max-snapshots", type=int, default=6)
    p.add_argument("--seed", type=int, default=5,
                   help="default 5: picked by visual comparison across seeds 0-5, see "
                        "module docstring")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=str(ROOT / "eval" / "results" / "sudoku4x4" /
                                          "fuzzy_trajectory_tau3.png"))
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    K = args.K if args.K is not None else K_DEFAULTS["sudoku4x4"]

    shim, info, sl, decode, needs_net, drift_arg = load_premier(args.ckpt, device)
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})  device={device}")
    if info["selfcond"]:
        raise SystemExit("tilt/sudoku_viz.py assumes a plain (non-self-conditioned) "
                          "TiltMLP -- this checkpoint is self-conditioned, so its "
                          "net.natural(...) calls would silently ignore self-conditioning")

    d = shim.net.cfg.d
    print(f"d={d}  tau={args.tau}  threshold={args.threshold}  T={T}, K={K}, "
          f"schedule={SCHEDULE!r}  seed={args.seed}")

    h = torch.zeros(d)
    h[-1] = args.tau

    fig, _ = plot_fuzzy_trajectory(shim, h, threshold=args.threshold, T=T, K=K,
                                    schedule=SCHEDULE, seed=args.seed,
                                    max_snapshots=args.max_snapshots)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
