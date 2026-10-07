"""Sudoku 9x9 check 3: memorization check -- exact match, plus nearest-neighbour
Hamming distance (MODEL vs. held-out training) for evidence that actually
discriminates when exact match can't.

Checkpoint : ckpt/sudoku9_v1_l17_d512x8_xf_sc_s8264_180k.pt
             (TransformerTilt, self-conditioned, d=325)
Reference  : data/sudoku9_v1's own valid rows (`GridBackedX.valid_grids()`,
             ~500,883 rows per meta.json -- read straight from the compact
             (n, 81) uint8 form, no bit materialisation/decode round-trip),
             split in half: one half is the REFERENCE set both MODEL and
             HELD are compared against, the other half (never used as a
             reference) IS the HELD population.

Two things are reported, not just one
--------------------------------------
1. EXACT match (all 81 cells) against the reference half -- see the
   "why exact match is still worth checking" note below for why this is a
   weak but asymmetric test (informative if it fires, silent if not).
2. NEAREST-NEIGHBOUR cell-Hamming distance from each MODEL sample to its
   closest reference grid, compared against the SAME quantity computed for
   HELD (genuine held-out training grids, never seen as a reference). This is
   what actually discriminates "generalised the constraint" from "memorised
   training rows": if MODEL's NN-distance distribution looks like HELD's, the
   model's valid outputs sit among the training manifold about as far from a
   given reference grid as another genuine unseen valid solution would --
   unsurprising, not evidence of copying. If MODEL's distances were
   systematically much SMALLER, that would suggest samples cluster
   suspiciously close to reference rows.
   (A third, `UNIFORM`-null population -- fresh draws from a true uniform
   sampler over all valid grids -- was considered and deliberately cut for
   this pass to keep the check quick; MODEL-vs-HELD is the comparison that
   most directly speaks to memorisation.)

Why exact match is still worth checking even though it's a weak test at this
scale: chance-level exact match against ~250,000 reference rows (half the
training set) out of 6,670,903,752,021,072,936,960 total valid grids is
~3.7e-15 % -- effectively zero -- so ANY exact match at all would be a strong
(not just suggestive) memorisation signal, even though the ABSENCE of one
proves little on its own (that's equally what perfect generalisation looks
like). It's a cheap, asymmetric tripwire: informative if it fires, silent if
not -- which is exactly why the NN-distance comparison above exists alongside
it, not instead of it.

self_cond_mode is explicit (see Appendix B.4 and the
`eval/sudoku9_selfcond_mode_check.py` finding that carry gives substantially
higher actual grid validity than two_pass on this checkpoint) -- default is
"carry" here since that's the mode reported for 9x9 in
the paper, but two_pass can be run for comparison too.

Usage:
    python eval/sudoku9_check3_memorization.py
    python eval/sudoku9_check3_memorization.py --self-cond-mode two_pass --tau 3.0 --n 1000
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
from common.sample_config import K_DEFAULTS, SCHEDULE, T  # noqa: E402

from tilt.data import load_dataset  # noqa: E402
from tilt.sudoku9 import D, N_DIGITS, TOTAL_VALID_SOLUTIONS, decode_grid_batch, is_valid_sudoku9_batch  # noqa: E402

CKPT = ROOT / "ckpt" / "sudoku9_v1_l17_d512x8_xf_sc_s8264_180k.pt"
DATA = ROOT / "data" / "sudoku9_v1"
ROW_NOISE = 0.02
P_INJECT = 0.05
CELLS = 81


def flat_codes(grids: torch.Tensor) -> np.ndarray:
    """(N, 9, 9) decoded grid -> (N, 81) int8, the comparison form."""
    return grids.cpu().numpy().reshape(len(grids), CELLS).astype(np.int8)


def _onehot(codes: np.ndarray) -> np.ndarray:
    """(N, 81) values 1..9 -> (N, 729) float32 one-hot, for the matmul below."""
    n = len(codes)
    out = np.zeros((n, CELLS * N_DIGITS), dtype=np.float32)
    idx = np.arange(CELLS) * N_DIGITS + (codes.astype(np.int64) - 1)
    np.put_along_axis(out, idx, 1.0, axis=1)
    return out


def min_hamming(query: np.ndarray, ref: np.ndarray, ref_chunk: int = 20_000) -> np.ndarray:
    """Exact minimum cell-Hamming distance from each query row to any reference row.

    Counts matching cells as a one-hot inner product (81 - Q @ R.T), chunked
    over the reference set to bound memory -- same method as
    `scripts/run_sudoku9_memorization.py::min_hamming`, reimplemented here so
    this eval script stays self-contained rather than importing from scripts/.
    """
    Q = _onehot(query)
    best = np.full(len(query), CELLS, dtype=np.int16)
    for i in range(0, len(ref), ref_chunk):
        R = _onehot(ref[i:i + ref_chunk])
        matches = Q @ R.T
        best = np.minimum(best, (CELLS - matches.max(axis=1)).astype(np.int16))
    return best


def training_split(datadir, max_ref: int = 250_000, seed: int = 0):
    """Row-level split of the training set's valid rows into (ref, held).

    Splits BEFORE any deduplication -- if the dataset happened to contain
    genuinely repeated valid grids, this is what would let them land on both
    sides (a bug in `scripts/run_sudoku9_memorization.py`'s `np.unique()`-first
    approach made this structurally impossible there). In practice
    `data/sudoku9_v1`'s ~500,883 valid rows are already 100% mutually distinct
    (meta.json: distinct_valid == n_valid), so this fix doesn't change the exact-match rate for THIS
    dataset -- it's included for correctness and consistency with the 6x6
    methodology, not because it's expected to matter here.
    """
    x, _frozen, _t, _meta = load_dataset(datadir)
    if hasattr(x, "valid_grids"):
        codes = x.valid_grids().astype(np.int8)   # (n_valid, 81), NOT deduped
        n_rows = x.n
    else:
        grids = decode_grid_batch(x[:, :D])
        ok = is_valid_sudoku9_batch(grids).cpu().numpy().astype(bool)
        codes = flat_codes(grids[ok])
        n_rows = len(x)
    n_valid = len(codes)
    n_distinct = len(np.unique(codes, axis=0))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_valid)
    half = n_valid // 2
    ref_codes = codes[perm[:half]][:max_ref]
    held_codes = codes[perm[half:]]
    return ref_codes, held_codes, n_rows, n_valid, n_distinct


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--self-cond-mode", default="carry", choices=["two_pass", "carry"])
    p.add_argument("--tau", type=float, default=3.0)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--row-noise", type=float, default=ROW_NOISE)
    p.add_argument("--p-inject", type=float, default=P_INJECT)
    p.add_argument("--K", type=int, default=None,
                   help="defaults to K_DEFAULTS['sudoku9']")
    p.add_argument("--max-ref", type=int, default=250_000,
                   help="cap on the reference half, to bound the NN-distance cost")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "sudoku9"))
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    K = args.K if args.K is not None else K_DEFAULTS["sudoku9"]

    print("=== training reference/held split ===")
    ref_codes, held_codes, n_rows, n_valid, n_distinct = training_split(
        args.data, max_ref=args.max_ref, seed=args.seed)
    ref_set = {c.tobytes() for c in ref_codes}
    print(f"  {n_rows:,} rows, {n_valid:,} valid ({100 * n_valid / n_rows:.3f}%), "
          f"{n_distinct:,} distinct overall (covers {100 * n_distinct / TOTAL_VALID_SOLUTIONS:.2e}% "
          f"of all {TOTAL_VALID_SOLUTIONS:,} valid grids)")
    print(f"  split: {len(ref_codes):,} reference rows, {len(held_codes):,} held-out rows "
          f"(row-level split, before any deduplication)")
    chance = len(ref_set) / TOTAL_VALID_SOLUTIONS
    print(f"  chance-level exact match per sample vs. the reference half: {100 * chance:.3e} %  "
          f"(expected count in {args.n} draws: {args.n * chance:.2e})")

    print("\n=== sampling MODEL ===")
    shim, info, sl, decode, needs_net, drift_arg = load_premier(args.ckpt, device)
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})  "
          f"self_cond_mode={args.self_cond_mode!r}  device={device}")
    d = shim.net.cfg.d
    h0 = torch.zeros(d)
    h0[-1] = args.tau
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        raw = sl(drift_arg, d=d, n=args.n, h=h0, T=T, K=K, schedule=SCHEDULE,
                 seed=args.seed, self_cond_mode=args.self_cond_mode)
    hard = decode(raw, drift_arg, h0)
    core = hard[:, :D]
    grids = decode_grid_batch(core)
    valid_mask = is_valid_sudoku9_batch(core).cpu().numpy().astype(bool)
    codes = flat_codes(grids)
    model_codes = codes[valid_mask]                       # only valid samples are comparable

    is_exact_match = np.fromiter((c.tobytes() in ref_set for c in codes), bool, len(codes))
    n_valid_gen = int(valid_mask.sum())
    n_matches = int(is_exact_match.sum())

    print(f"\n=== exact-match result: tau={args.tau}, n={args.n}, K={K}, "
          f"self_cond_mode={args.self_cond_mode!r} ===")
    print(f"  {n_valid_gen}/{args.n} samples decoded to a valid grid ({100 * n_valid_gen / args.n:.2f}%)")
    print(f"  {n_matches}/{args.n} samples EXACTLY match a reference-half training row "
          f"({100 * n_matches / args.n:.4f}%)")
    if n_matches == 0:
        print("  no exact matches -- consistent with (but does not prove) no memorisation; "
              "at this reference-set coverage, chance alone predicts "
              f"{args.n * chance:.2e} matches in expectation, so 0 is unsurprising either way.")
    else:
        print(f"  {n_matches} exact match(es) against a chance expectation of "
              f"{args.n * chance:.2e} -- given how small that expectation is, ANY match here "
              f"is a strong signal, not a coincidence.")

    print("\n=== nearest-neighbour Hamming distance: MODEL vs. HELD-OUT training ===")
    n_cmp = min(len(model_codes), len(held_codes))
    held_sample = held_codes[:n_cmp]
    nn_model = min_hamming(model_codes, ref_codes)
    nn_held = min_hamming(held_sample, ref_codes)
    print(f"  MODEL  n={len(model_codes):5d}  NN median={np.median(nn_model):5.1f}  "
          f"p05={np.percentile(nn_model, 5):5.1f}  min={nn_model.min():3d}  mean={nn_model.mean():5.2f}")
    print(f"  HELD   n={len(held_sample):5d}  NN median={np.median(nn_held):5.1f}  "
          f"p05={np.percentile(nn_held, 5):5.1f}  min={nn_held.min():3d}  mean={nn_held.mean():5.2f}")
    print("  MODEL ~ HELD -> the model's valid outputs sit among the training manifold about "
          "as far from a\n  reference grid as another genuine unseen valid solution does -- "
          "not evidence of memorisation.\n  MODEL << HELD (much smaller distances) would instead "
          "suggest suspicious clustering near the reference set.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "ckpt": Path(args.ckpt).name, "self_cond_mode": args.self_cond_mode,
        "tau": args.tau, "n": args.n, "K": K,
        "n_valid_generated": n_valid_gen, "n_exact_matches": n_matches,
        "chance_expected_matches": args.n * chance,
        "n_reference_rows": len(ref_codes), "n_held_rows": len(held_codes),
        "nn_model_median": float(np.median(nn_model)), "nn_model_p05": float(np.percentile(nn_model, 5)),
        "nn_model_min": int(nn_model.min()), "nn_model_mean": float(nn_model.mean()),
        "nn_held_median": float(np.median(nn_held)), "nn_held_p05": float(np.percentile(nn_held, 5)),
        "nn_held_min": int(nn_held.min()), "nn_held_mean": float(nn_held.mean()),
    }
    out_csv = out_dir / f"check3_memorization_{args.self_cond_mode}_tau{args.tau:g}.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False)

    # Raw distances saved too (not just the summary CSV above), so a later
    # run can regenerate/restyle plots without re-sampling from the
    # checkpoint or re-running the CP-SAT-free but still nontrivial NN solve.
    nn_path = out_dir / f"check3_memorization_{args.self_cond_mode}_tau{args.tau:g}_nn_distances.npz"
    np.savez(nn_path, nn_model=nn_model, nn_held=nn_held)

    bins = np.arange(min(nn_model.min(), nn_held.min()) - 0.5,
                      max(nn_model.max(), nn_held.max()) + 1.5, 1.0)
    fig, (ax_cdf, ax_hist) = plt.subplots(1, 2, figsize=(13, 5))
    for name, nn, color in (("MODEL", nn_model, "tab:orange"), ("HELD-OUT training", nn_held, "tab:blue")):
        dist = np.sort(nn)
        cdf = np.arange(1, len(dist) + 1) / len(dist)
        ax_cdf.step(dist, cdf, where="post", label=f"{name} (n={len(nn)})", color=color)
        ax_hist.hist(nn, bins=bins, histtype="step", linewidth=2, density=True,
                      label=f"{name} (n={len(nn)})", color=color)
    ax_cdf.set_xlabel("nearest-neighbour cell-Hamming distance to reference set")
    ax_cdf.set_ylabel("cumulative fraction of samples")
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.grid(alpha=0.3)
    ax_cdf.legend(fontsize=9)
    ax_cdf.set_title("CDF")

    ax_hist.set_xlabel("nearest-neighbour cell-Hamming distance to reference set")
    ax_hist.set_ylabel("density")
    ax_hist.grid(alpha=0.3)
    ax_hist.legend(fontsize=9)
    ax_hist.set_title("histogram (density-normalized, shared bins)")

    fig.suptitle(f"Sudoku 9x9 memorization check: MODEL vs. HELD-OUT  "
                 f"(tau={args.tau}, self_cond_mode={args.self_cond_mode!r})")
    fig.tight_layout()
    out_png = out_dir / f"check3_memorization_{args.self_cond_mode}_tau{args.tau:g}_nn_cdf.png"
    fig.savefig(out_png, dpi=150)

    print(f"\nsaved: {out_csv}")
    print(f"saved: {nn_path}")
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
