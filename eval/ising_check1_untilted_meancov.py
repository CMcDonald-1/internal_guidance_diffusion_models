"""Ising check 1: untilted (h=0) mean vector / covariance comparison,
exact ground-truth draws vs. network-sampled draws.

Checkpoint : ckpt/ising12_exact10_lambda3p5_30k_jac.pt
             (plain TiltMLP, trained on data/run_exact10)
Dataset    : data/run_exact10  (d=13: IsingTarget(d=12) reward-augmented
             with reward="exact:10" -> d=13, exact enumeration available)

Sampling config (see eval/common/sample_config.py -- the shared, per-problem
canonical (T, K, schedule) so this stays consistent across every check/
problem going forward): T=20.0, K=K_DEFAULTS["ising"]=1000, schedule="sstar".

What this check does:
    1. Draw N samples directly from the target's own untilted law pi (h=0),
       via target.sample(n) -- these are exact iid draws, no SDE involved.
    2. Draw N samples via the trained network's sampler (sl_sample +
       decode_drift), also at h=0 (no reward tilt).
    3. Compare the two empirical (N-sample) mean vectors and covariance
       matrices: SSR and Frobenius norm of the difference, plus a raster
       (imshow heatmap) plot of both covariance matrices and their
       difference, side by side.

Usage:
    python eval/ising_check1_untilted_meancov.py --n 20000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # standalone subprocess run: no GUI backend available
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.checkpoints import load_premier  # noqa: E402
from common.sample_config import K_DEFAULTS, SCHEDULE, T  # noqa: E402

from tilt.data import load_dataset  # noqa: E402

CKPT = ROOT / "ckpt" / "ising12_exact10_lambda3p5_30k_jac.pt"
DATA = ROOT / "data" / "run_exact10"


def sample_mean_cov(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False, ddof=1)
    return mean, cov


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "ising"))
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    K = K_DEFAULTS["ising"]

    _x_train, _frozen, target, _meta = load_dataset(args.data)
    if target is None:
        raise SystemExit(f"{args.data} has no target.json -- exact ground truth unavailable")

    _shim, info, sl, decode, _needs_net, drift_arg = load_premier(args.ckpt, device)
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})  device={device}")
    print(f"sampling config: T={T}, K={K}, schedule={SCHEDULE!r}, n={args.n}, d={target.d}")

    # 1. exact ground-truth draws -- literally iid from pi, no SDE involved.
    exact_draws = target.sample(
        args.n, generator=torch.Generator().manual_seed(args.seed)
    ).numpy()

    # 2. network-sampled draws, h=0 (untilted).
    raw = sl(drift_arg, d=target.d, n=args.n, h=None, T=T, K=K,
              schedule=SCHEDULE, seed=args.seed + 1)
    net_draws = decode(raw, drift_arg, None).numpy()

    mean_exact, cov_exact = sample_mean_cov(exact_draws)
    mean_net, cov_net = sample_mean_cov(net_draws)

    mean_diff = mean_net - mean_exact
    cov_diff = cov_net - cov_exact

    ssr_mean = float(np.sum(mean_diff**2))
    ssr_cov = float(np.sum(cov_diff**2))
    frob_cov = float(np.sqrt(ssr_cov))

    print("\n-- mean vector (net - exact) --")
    print(f"  SSR  = {ssr_mean:.6f}")
    print(f"  max|diff| = {np.max(np.abs(mean_diff)):.6f}")

    print("\n-- covariance matrix (net - exact) --")
    print(f"  SSR              = {ssr_cov:.6f}")
    print(f"  Frobenius norm   = {frob_cov:.6f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "check1_untilted_meancov.npz",
        mean_exact=mean_exact, mean_net=mean_net,
        cov_exact=cov_exact, cov_net=cov_net,
        ssr_mean=ssr_mean, ssr_cov=ssr_cov, frob_cov=frob_cov,
        n=args.n, seed=args.seed,
    )

    vmax_cov = max(np.abs(cov_exact).max(), np.abs(cov_net).max())
    vmax_diff = np.abs(cov_diff).max()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, mat, title, vmax in zip(
        axes,
        [cov_exact, cov_net, cov_diff],
        ["exact (ground truth)", "network sampler", "net - exact"],
        [vmax_cov, vmax_cov, vmax_diff],
    ):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("coord")
        ax.set_ylabel("coord")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"Ising d={target.d} untilted covariance, n={args.n} "
        f"(SSR={ssr_cov:.4f}, Frob={frob_cov:.4f})"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "check1_untilted_cov_raster.png", dpi=150)

    print(f"\nsaved: {out_dir / 'check1_untilted_meancov.npz'}")
    print(f"saved: {out_dir / 'check1_untilted_cov_raster.png'}")


if __name__ == "__main__":
    main()
