# Internal Guidance as a Sampling Method for Tilted Diffusion Models — code

This repository contains the code needed to reproduce every empirical result in the
paper: the target densities, model definitions, dataset generators, training
scripts, and — most importantly — the **evaluation scripts in `eval/`**, which
regenerate the figures and tables from the trained checkpoints.

Datasets (~1 GB) and trained checkpoints (~194 MB) are too large to include here
and are hosted at:

> **https://osf.io/kewg9/**

The quickest route is: install → download → run `eval/` (Sections 1–3).
Retraining from scratch is optional (Section 5).

---

## Repository layout

```
eval/        evaluation scripts (the main entry point) + reference outputs in eval/results/
  common/    shared helpers: checkpoint loading, sampler settings (T, K, schedule), CI utilities
tilt/        library code: target densities, noising, models (MLP, self-conditioned MLP,
             transformer), training loops, stochastic-localization samplers, Sudoku/MNIST encodings
scripts/     dataset generators and training entry points
hpc/         Slurm scripts used for the GPU training runs (6x6, 9x9, MNIST) and dataset builds
data/        <- put the downloaded datasets here   (see data/README.md)
ckpt/        <- put the downloaded checkpoints here (see ckpt/README.md)
```

All commands below are run from the repository root.

## 1. Install

Python 3.11+. For GPU runs, first install a CUDA build of PyTorch following
https://pytorch.org/get-started/locally/ (the pinned `torch==2.11.0` then counts as
already satisfied). Then:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is recommended for the 6x6, 9x9 and MNIST evaluations; the Ising and
4x4 Sudoku evaluations run on CPU in minutes. Every eval script picks CUDA
automatically when available and accepts `--device cpu|cuda`.

## 2. Download datasets and checkpoints

From the data link above, download the dataset archives and the checkpoints, then
arrange them so the paths below exist (unzip each dataset into `data/`):

```
data/run_exact10/        meta.json  target.json  train.csv           (Ising, d=12)
data/sudoku_reward_v2/   meta.json  target.json  train.csv           (Sudoku 4x4)
data/sudoku6_v2/         meta.json  grids.npy    reward.npy          (Sudoku 6x6)
data/sudoku9_v1/         meta.json  grids.npy    reward.npy          (Sudoku 9x9)
data/mnist_v1/           meta.json  images.npy   labels.npy  holdout_images.npy  holdout_labels.npy

ckpt/ising12_exact10_lambda3p5_30k_jac.pt
ckpt/sudoku_reward_v2_infill_mode1p25_lambda5_w1024b8_30k.pt
ckpt/sudoku6_v2_l10_xf_sc_s8264_180k.pt
ckpt/sudoku9_v1_l17_d512x8_xf_sc_s8264_180k.pt
ckpt/mnist_v1_l24_d256x4_xf_sc_s8264_30k.pt
ckpt/mnist_cnn_scorer.pt                       (independent CNN used only to score MNIST samples)
```

Each checkpoint stores its own architecture and training hyperparameters
(`cfg` and `extra` entries), and each dataset's `meta.json` records its seed,
size and generation parameters.

## 3. Run the evaluations

Each script is standalone and writes CSV + PNG output to `eval/results/<problem>/`.
**Reference outputs from the paper are already in `eval/results/`; rerunning a
script overwrites them**, so copy that folder first if you want to compare
(`cp -r eval/results eval/results_paper`).

All samplers use the paper's settings (T = 20, K = 1000 Euler steps, the
schedule of Appendix B.5), fixed in `eval/common/sample_config.py`.

| Paper result | Command |
|---|---|
| **Ising (Sec. 5.1, App. C.1)** | |
| Fig. 7 — covariance, exact vs. network | `python eval/ising_check1_untilted_meancov.py` |
| Fig. 8 — denoiser L2 error vs. signal level and τ | `python eval/ising_check2_denoiser_ood.py` |
| Fig. 9 — reward log-odds vs. τ | `python eval/ising_check3_reward_logodds.py` |
| **Sudoku 4x4 (Sec. 5.2)** | |
| Fig. 11 — denoiser L2 error | `python eval/sudoku4x4_check1_denoiser_ood.py` |
| Fig. 12 — log-odds (network and exact-denoiser samplers) | `python eval/sudoku4x4_check2_reward_logodds.py` |
| Fig. 13 — early bit-by-bit decoding trajectory | `python eval/sudoku4x4_fuzzy_trajectory_figure.py` |
| Fig. 14 — true vs. network denoiser at a = 0.5 | `python eval/sudoku4x4_lollipop_a0p5.py` |
| Fig. 15 — χ² uniformity over the 288 solutions | `python eval/sudoku4x4_check3_chisq_uniform.py --tau 4` |
| **Sudoku 6x6 (Sec. 5.3)** | |
| Fig. 17 — denoiser L2 error at τ = 0 | `python eval/sudoku6_check1_denoiser_ood.py` |
| Fig. 18 — log-odds | `python eval/sudoku6_check2_reward_logodds.py` |
| Memorization (10 / 1000 exact matches) | `python eval/sudoku6_check3_memorization.py` |
| **Sudoku 9x9 (Sec. 5.4)** | |
| Fig. 20a / 20b — log-odds, two-pass / carry self-conditioning | `python eval/sudoku9_check2_reward_logodds.py --self-cond-mode two_pass` (and `carry`) |
| Fig. 21 — distance to nearest valid grid | `python eval/sudoku9_nearest_puzzle_cdf.py`, then `..._buckets.py`, `..._pmf.py` (these two re-plot the CSV) |
| Fig. 23 — memorization (nearest-neighbour distance) | `python eval/sudoku9_check3_memorization.py` |
| Fig. 4 / Table 1 — infilling vs. number of given cells | `python eval/sudoku9_check4_infill.py` |
| Fig. 3 — example completions | `python eval/sudoku9_infill_figure.py` (conditional) and `--n-removed 81` (unconditional) |
| **MNIST (Sec. 5.5)** | |
| Figs. 24–27 / Tables 2–3 — class log-odds | `python eval/mnist_check1_reward_logodds.py --self-cond-mode two_pass` (and `carry`) |
| Fig. 28 — held-out vs. generated digits | `python eval/mnist_sample_figure.py --cls 0 --real`, then `--cls 0 --self-cond-mode two_pass` / `carry` |

Supporting checks referred to in the appendix (Euler-step sweeps and the
self-conditioning comparison): `eval/sudoku6_k_sweep.py`, `eval/sudoku9_k_sweep.py`,
`eval/sudoku9_selfcond_mode_check.py`, `eval/mnist_k_sweep.py`, `eval/mnist_denoiser_norm.py`.
See `eval/README.md` for what each script computes and its options.

**Self-conditioning mode.** The two sampling modes of Appendix B.4 are selected
with `--self-cond-mode`. The paper reports both; `carry` is used for the 9x9
infilling and memorization results, and `two_pass` is the default elsewhere.
`carry` prints a warning because its sampled chain is not Markov (Appendix B.4).

**MNIST scorer.** If `ckpt/mnist_cnn_scorer.pt` is absent, the MNIST scripts train
a fresh scorer on first use (3 epochs, seed 0; a few minutes).

## 4. Hardware used for the paper

The 6x6, 9x9 and MNIST models were each trained on a single NVIDIA H100 80GB GPU;
training times and compute estimates for every run are given in Appendix C. The
Ising and 4x4 models are small (88K and 17M parameters) and train on any single GPU.

## 5. Regenerating datasets and retraining (optional)

The hosted datasets are the exact files used in the paper. The commands below
rebuild datasets **from the same distributions**; because random draws depend on
the PyTorch version, regenerated files are statistically equivalent rather than
byte-identical. Ising note: the hosted `data/run_exact10/target.json` stores the
exact coupling matrix J and field s used (J0 = 1.5, σ = 0.3).

### Datasets

```bash
# Ising d=12, reward = exactly 10 of 12 spins positive
python scripts/make_data.py --kind ising --d 12 --J0 1.5 --field 0.3 --reward exact:10 \
    -n 100000 --seed 2417 --out data/run_exact10
# Sudoku 4x4
python scripts/make_data.py --kind sudoku --d 33 --reward sudoku_valid --row-noise 0.02 \
    --p-inject 0.05 -n 150000 --seed 4821 --out data/sudoku_reward_v2
# Sudoku 6x6 (5M rows) and 9x9 (10M rows); CPU-bound, a few minutes each
python scripts/make_sudoku6_data.py --n 5000000  --outdir data/sudoku6_v2 --seed 5209 \
    --row-noise 0.02 --p-inject 0.05 --format npy
python scripts/make_sudoku9_data.py --n 10000000 --outdir data/sudoku9_v1 --seed 5209 \
    --row-noise 0.02 --p-inject 0.05 --gen rowmajor --format npy
# MNIST (downloads the standard IDX files, ~11 MB)
python scripts/make_mnist_data.py --out data/mnist_v1
```

### Training

Ising and 4x4 Sudoku (residual MLP; minutes to an hour on one GPU):

```bash
python scripts/run_train.py --data data/run_exact10 --out ckpt/ising12_exact10_lambda3p5_30k_jac.pt \
    --steps 30000 --batch-size 1024 --lambda-max 3.5 --seed 0
python scripts/run_train.py --data data/sudoku_reward_v2 \
    --out ckpt/sudoku_reward_v2_infill_mode1p25_lambda5_w1024b8_30k.pt \
    --width 1024 --blocks 8 --steps 30000 --batch-size 8192 \
    --lambda-max 5 --mode1-prob 0.25 --lambda1-prob 0.25 --seed 7
```

6x6, 9x9 and MNIST (self-conditioned transformer) were trained with the Slurm
scripts in `hpc/`. Set `--account` / `--partition` in each script for your
cluster, run `bash hpc/setup_env.sh` once, then submit from the repository root:

```bash
sbatch --array=0-0 -t 05:00:00 hpc/train_sudoku6_v2.sbatch xf_sc
sbatch --array=0-0 -t 30:00:00 --export=ALL,D_MODEL=512,N_LAYERS=8,TAG=d512x8 \
    hpc/train_sudoku9_v1.sbatch xf_sc
sbatch --array=0-0 -t 02:30:00 hpc/train_mnist_v1.sbatch xf_sc
```

Array index 0 selects seed 8264, the seed of every reported transformer run.
The scripts write resumable partial checkpoints, so a run killed by the
wall-clock limit can be resubmitted unchanged and continues where it stopped.
Without Slurm, the same runs can be launched directly with
`scripts/run_train_sudoku6_reward.py`, `run_train_sudoku9_reward.py` and
`run_train_mnist_reward.py`; the exact arguments are visible in the corresponding
`.sbatch` file.
