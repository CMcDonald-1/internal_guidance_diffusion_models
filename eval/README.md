# Evaluation scripts

Every script here is standalone: run it from the repository root as
`python eval/<script>.py [options]`. It loads its checkpoint from `ckpt/` and
(where needed) its dataset from `data/`, and writes CSV + PNG files to
`results/<problem>/` inside this folder. `--help` lists every option.

Shared code in `common/`:

- `checkpoints.py` — loads a checkpoint (architecture is read from the file) and
  runs the network on GPU while the sampler state stays on CPU.
- `sample_config.py` — the sampler settings used throughout the paper:
  T = 20, K = 1000 Euler steps, schedule "sstar" (Appendix B.5).
- `reward_check.py` — exact reward rates, Clopper–Pearson intervals and the
  τ grid anchored at τ98 (τ95 for MNIST).

The four kinds of check:

1. **Reward log-odds** — sample at tilt strength τ and compare the rate of
   reward-achieving samples with the exact log-odds curve (Appendix B.3).
2. **Denoiser vs. ground truth** — L2 error of the network denoiser against the
   exact one; only where the exact denoiser is tractable (Ising, 4x4, and the
   exact row marginal for 6x6).
3. **Memorization** — exact matches and nearest-neighbour distances between
   generated samples and the training set.
4. **Sample quality** — problem-specific (uniformity over 4x4 solutions,
   distance to the nearest valid 9x9 grid, 9x9 infilling).

## Scripts

| Script | What it computes |
|---|---|
| `ising_check1_untilted_meancov.py` | Untilted mean and covariance, exact vs. network samples (Fig. 7). |
| `ising_check2_denoiser_ood.py` | Denoiser SSE across signal levels a and tilts τ, fresh samples per grid point (Fig. 8). |
| `ising_check3_reward_logodds.py` | Reward log-odds vs. τ with 95% CIs (Fig. 9). |
| `sudoku4x4_check1_denoiser_ood.py` | Denoiser SSE vs. exact operator across a and τ (Fig. 11). |
| `sudoku4x4_check2_reward_logodds.py` | Log-odds for the network sampler and for the same sampler driven by the exact denoiser, separating discretization from network error (Fig. 12). `--K` reruns at another step count. |
| `sudoku4x4_fuzzy_trajectory_figure.py` | Coordinate trajectories and early bit-by-bit decoding along one sampling path (Fig. 13). |
| `sudoku4x4_lollipop_a0p5.py` | True vs. network denoiser outputs at a = 0.5 for τ = 0 and 4 (Fig. 14). |
| `sudoku4x4_check3_chisq_uniform.py` | χ² test that valid samples are uniform over the 288 solutions, network and exact-denoiser samplers (Fig. 15; use `--tau 4`). |
| `sudoku6_check1_denoiser_ood.py` | Denoiser error vs. the exact row marginal at τ = 0 (Fig. 17). |
| `sudoku6_check2_reward_logodds.py` | Log-odds vs. the closed-form success rate (Fig. 18). |
| `sudoku6_check3_memorization.py` | Exact matches of 1000 samples at τ98 against all 5M training rows. |
| `sudoku6_k_sweep.py` | Validity vs. Euler step count K. |
| `sudoku9_check2_reward_logodds.py` | Log-odds of the model's reward bit and of actual grid validity; `--self-cond-mode two_pass` or `carry` is required (Fig. 20). |
| `sudoku9_nearest_puzzle_cdf.py` | Exact (CP-SAT) bit distance from each sample to the nearest valid grid, CDF (Fig. 21). `--replot-only` redraws from the CSV. |
| `sudoku9_nearest_puzzle_buckets.py`, `sudoku9_nearest_puzzle_pmf.py` | Bucketed and per-distance views of the same CSV (no sampling). |
| `sudoku9_check3_memorization.py` | Exact matches and nearest-neighbour distance vs. a held-out split of the training grids (Fig. 23). |
| `sudoku9_check4_infill.py` | Infilling: P(valid), P(given cells kept) and their joint vs. number of given cells (Fig. 4, Table 1). |
| `sudoku9_infill_figure.py` | One example completion; `--n-removed 81` gives an unconditional sample (Fig. 3). |
| `sudoku9_k_sweep.py`, `sudoku9_selfcond_mode_check.py` | Euler-step sweep and a direct two-pass vs. carry comparison on the same checkpoint. |
| `mnist_check1_reward_logodds.py` | Class log-odds for all 10 classes, scored by the CNN with its measured per-class error rates; `--self-cond-mode` required (Figs. 24–27, Tables 2–3). `--replot-only` redraws from the CSV. |
| `mnist_sample_figure.py` | Annotated 5x5 sample grid for one class; `--real` draws held-out digits through the same plotting path (Fig. 28). |
| `mnist_k_sweep.py`, `mnist_denoiser_norm.py` | Euler-step sweep and the denoiser-norm diagnostic along the sampling path. |

## Notes

- Sampling is seeded: given (seed, n, K, T, schedule, self-conditioning mode),
  reruns with the same PyTorch version and device should closely reproduce the
  CSVs in `results/` (GPU kernels can introduce small run-to-run differences).
- `results/` ships with the outputs used in the paper; rerunning a script
  overwrites its files.
