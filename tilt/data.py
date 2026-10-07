"""Training-set generation, CSV persistence, and batch construction.

Design note.  What gets written to disk is the set of samples X ~ pi, NOT the
noised pairs.  Samples from pi are the expensive artifact (enumeration here,
MCMC at scale); the Gaussian noise is free.  Freezing (X, U, a) triples into a
file means training on one fixed noise realisation, which lets the network
overfit the noise instead of learning the operator.  Use freeze_noise=True only
when you want bit-exact reproducibility and understand the cost.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .noising import (
    a_of_beta,
    profile_diag,
    sample_beta_loguniform,
    sample_infill_q,
    vp_noise,
)
from .targets import EnumerableTarget


def make_dataset(target: EnumerableTarget, n: int, outdir: str | Path,
                 seed: int = 0, freeze_noise: bool = False,
                 beta_lo: float = 0.02, beta_hi: float = 50.0) -> Path:
    """Draw n exact samples from pi and write them to <outdir>/train.csv."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)

    x = target.sample(n, generator=g)
    cols = {f"x{i}": x[:, i].numpy() for i in range(target.d)}

    if freeze_noise:
        beta = sample_beta_loguniform(n, beta_lo, beta_hi, generator=g)
        a = a_of_beta(beta)
        u = vp_noise(x, a, generator=g)
        cols.update({f"u{i}": u[:, i].numpy() for i in range(target.d)})
        cols["a"] = a[:, 0].numpy()
        cols["beta"] = beta[:, 0].numpy()

    pd.DataFrame(cols).to_csv(outdir / "train.csv", index=False)
    target.save(outdir / "target.json")
    meta = {
        "n": n, "d": target.d, "seed": seed, "freeze_noise": freeze_noise,
        "beta_lo": beta_lo, "beta_hi": beta_hi,
        "effective_support": target.effective_support(),
    }
    if hasattr(target, "success_rate"):
        meta["success_rate"] = target.success_rate()
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    return outdir / "train.csv"


class TokenBackedX:
    """A lazy stand-in for the dense (n, d) training tensor, backed by token ids.

    Why this exists: `load_dataset` used to return a materialised (n, d)
    float32 tensor read from train.csv. That is fine at n=100k (60 MB) and
    impossible at n=30M -- (30e6, 151) float32 is 18 GB resident, and the CSV
    it came from would be 20 GB of text. But the rhyme_gpt2 domain's 150 core
    bits are a deterministic function of 10 token ids, so the compact form
    (n, 10) uint16 + (n,) int8 is 21 bytes/row (630 MB at 30M, small enough to
    sit in RAM) and loses nothing.

    This class materialises bits for the SAMPLED ROWS ONLY. It exposes exactly
    the two things `BatchSampler` uses -- `.shape` and `__getitem__` with an
    index tensor -- so BatchSampler, train(), and every downstream caller work
    unmodified; the dense path is still used for every other dataset.

    Layout of the returned rows matches make_rhyme_dataset.py exactly:
    coords 0..149 are token bits ({0,1} -> {-1,+1}, token-major, MSB-first
    within each 15-bit code) and coord 150 is the rhyme reward in {-1,+1}.
    """

    def __init__(self, tokens, reward, bits_by_token_id):
        self.tokens = tokens                       # (n, 10) uint16 (memmap ok)
        self.reward = reward                       # (n,) int8 in {-1,+1}
        self.bits = bits_by_token_id               # (32768, 15) uint8 in {0,1}
        self.n = int(tokens.shape[0])
        self.seq_len, self.n_bits = int(tokens.shape[1]), int(bits_by_token_id.shape[1])
        self.d = self.seq_len * self.n_bits + 1
        self.shape = (self.n, self.d)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx) -> torch.Tensor:
        i = idx.cpu().numpy() if torch.is_tensor(idx) else np.asarray(idx)
        tk = np.asarray(self.tokens[i])                      # (b, seq_len)
        b = self.bits[tk].reshape(len(i), -1)                # (b, seq_len*n_bits)
        out = np.empty((len(i), self.d), dtype=np.float32)
        out[:, : self.d - 1] = b.astype(np.float32) * 2.0 - 1.0
        out[:, self.d - 1] = self.reward[i]
        return torch.from_numpy(out)

    def dense(self) -> torch.Tensor:
        """Materialise the whole thing. Only for small n -- 18 GB at n=30M."""
        return self[np.arange(self.n)]


class StreamCropX:
    """Random 32-token windows cut live out of one flat token stream.

    Why this exists: TokenBackedX materialises a FIXED set of rows, which was
    right when the rows were expensive (each one a GPT-2 generation). For a real
    corpus the rows are free -- OpenWebText is 9.04e9 tokens, i.e. ~8.8e9 legal
    windows, against the 4.10e8 a 100k x 4096 run draws. Freezing 10M of them
    would throw away 99.5% of the corpus and reintroduce epochs (and with them
    the whole memorisation apparatus) for no reason.

    LEGALITY. A window is legal iff it contains no <|endoftext|>, i.e. it does
    not span two unrelated documents -- that transition is genuine garbage and
    not structure worth learning. There is no vocabulary filter: owt_v1 uses a
    16-bit codebook over the full 50,257 ids precisely so the 32,768-id
    restriction (which discards ~68% of real windows at length 32, and conditions
    the survivors on lexical commonness) can be dropped.

    The test is two binary searches against the sorted EOT positions,

        legal(i)  <=>  searchsorted(eot, i+L) == searchsorted(eot, i)

    which is why only the EOT positions are stored (8.0e6 int64 = 64 MB) rather
    than the legal offsets themselves (~8.8e9 int64 = 70 GB).

    THE INDEX IS A COUNT, NOT AN ADDRESS. `__getitem__` uses only `len(idx)`: it
    draws that many offsets uniformly, rejects the illegal ones and redraws,
    which is an exact uniform draw over legal windows. It does NOT return "row
    idx[k]", and calling it twice with the same index gives different windows.
    That is fine for the one caller that matters -- BatchSampler draws fresh
    uniform indices every step and never relies on the mapping -- but it means
    this class must not be used anywhere that treats X as an addressable array.
    Its own generator is seeded, so a run is still reproducible.

    Layout matches TokenBackedX: coords are token-major, MSB-first within each
    block, {0,1} -> {-1,+1}. There is no reward coordinate, so d = seq_len*bits.
    """

    def __init__(self, stream, eot_pos, bits_by_token_id, seq_len: int,
                 seed: int = 0, max_redraws: int = 100):
        self.stream = stream                       # (N,) uint16
        self.eot = np.asarray(eot_pos, dtype=np.int64)
        self.bits = bits_by_token_id               # (V, B) uint8 in {0,1}
        self.seq_len = int(seq_len)
        self.n_bits = int(bits_by_token_id.shape[1])
        self.d = self.seq_len * self.n_bits
        self.n = int(len(stream)) - self.seq_len
        if self.n <= 0:
            raise ValueError(f"stream of {len(stream)} tokens is shorter than "
                             f"seq_len={seq_len}")
        self.shape = (self.n, self.d)
        self.rng = np.random.default_rng(seed)
        self._ar = np.arange(self.seq_len, dtype=np.int64)
        self.max_redraws = max_redraws

    def __len__(self) -> int:
        return self.n

    def _legal(self, s: np.ndarray) -> np.ndarray:
        if len(self.eot) == 0:
            return np.ones(len(s), dtype=bool)
        return (np.searchsorted(self.eot, s + self.seq_len)
                == np.searchsorted(self.eot, s))

    def draw_offsets(self, m: int) -> np.ndarray:
        """m offsets drawn uniformly from the legal windows."""
        s = self.rng.integers(0, self.n, m)
        bad = ~self._legal(s)
        for _ in range(self.max_redraws):
            k = int(bad.sum())
            if k == 0:
                return s
            s[bad] = self.rng.integers(0, self.n, k)
            bad[bad] = ~self._legal(s[bad])
        raise RuntimeError(
            f"could not fill a batch of {m} legal windows in {self.max_redraws} "
            f"redraws ({int(bad.sum())} still illegal). Either eot_pos is wrong "
            f"or the documents are shorter than seq_len={self.seq_len}.")

    def __getitem__(self, idx) -> torch.Tensor:
        m = len(idx)
        s = self.draw_offsets(m)
        tk = np.asarray(self.stream[s[:, None] + self._ar]).astype(np.int64)
        b = self.bits[tk].reshape(m, -1)
        return torch.from_numpy(b.astype(np.float32) * 2.0 - 1.0)

    def acceptance(self, n_probe: int = 100_000) -> float:
        """Measured fraction of offsets that are legal -- a startup sanity print."""
        s = self.rng.integers(0, self.n, n_probe)
        return float(self._legal(s).mean())


def _load_bits_by_token_id(embed_dir: Path, flags: bool = False) -> np.ndarray:
    """(32768, 15) uint8 indexed by raw GPT-2 token id -- or (32768, 19) with flags.

    Inverts the sorted (binary_matrix, sorted_token_ids) pair written by
    text_based_test/gpt2_binary_embedding.py -- same inversion as
    make_rhyme_dataset.py::load_bits_by_token_id, duplicated here so tilt/
    does not import from text_based_test/.

    flags=True appends token_flags.npy (word_initial, 2-bit character class,
    function-word), making token_bits 19 instead of 15. Those bits are a
    DETERMINISTIC function of the token id, so they add no information about the
    sequence -- the bet is that they are far more context-predictable than token
    identity, giving the sampler an intermediate commitment the 15-bit code does
    not admit. They are appended, never interleaved, so coords [0:15] of each
    token block stay exactly what every existing decoder expects.
    """
    binary_matrix = np.load(embed_dir / "binary_matrix.npy")
    sorted_token_ids = np.load(embed_dir / "sorted_token_ids.npy")
    inv = np.empty(len(sorted_token_ids), dtype=np.int64)
    inv[sorted_token_ids] = np.arange(len(sorted_token_ids))
    bits = binary_matrix[inv]
    if not flags:
        return bits
    fp = embed_dir / "token_flags.npy"
    if not fp.exists():
        raise FileNotFoundError(
            f"TILT_TOKEN_FLAGS=1 but {fp} is missing. Build it with "
            f"python text_based_test/token_flags.py --embed-dir {embed_dir}")
    fl = np.load(fp)
    if len(fl) != len(bits):
        raise ValueError(f"token_flags has {len(fl)} rows, bits has {len(bits)}")
    print(f"[tilt.data] token flags ON: {bits.shape[1]} + {fl.shape[1]} = "
          f"{bits.shape[1] + fl.shape[1]} bits per token", flush=True)
    return np.concatenate([bits, fl.astype(bits.dtype)], axis=1)


def load_dataset(datadir: str | Path):
    """Return (X, frozen_or_None, target, meta).

    X is a dense (n, d) float32 tensor for CSV datasets, or a `TokenBackedX`
    view for token-backed ones (a directory containing tokens.npy/reward.npy,
    written by make_rhyme_dataset_v2.py + merge_rhyme_shards.py). Both support
    `.shape` and index-tensor `__getitem__`, which is all BatchSampler needs.

    target is None if <datadir>/target.json is absent -- this is the case for
    datasets built from an empirical/non-enumerable source (e.g. GPT-2
    generation) with no exact ground-truth operator available. Callers that
    need the target (e.g. evaluate.py's operator_error) must handle target=None
    themselves; tilt/train.py::train() never uses target during training, so
    it works unmodified in either case.
    """
    from .targets import EnumerableTarget as _T

    datadir = Path(datadir)
    target_path = datadir / "target.json"

    # --- stream-backed (corpus) datasets ----------------------------------
    # Written by scripts/make_owt_stream.py. No reward.npy and no fixed rows:
    # windows are cut live, so `d` is seq_len*token_bits with no +1.
    if (datadir / "stream.npy").exists():
        meta = json.loads((datadir / "meta.json").read_text())
        embed_dir = Path(meta["embed_dir"])
        if not embed_dir.is_absolute() and not embed_dir.exists():
            embed_dir = datadir.parent.parent / meta["embed_dir"]

        # Same reasoning as the token-backed path: BatchSampler draws a fresh set
        # of random offsets every step, so a memmap turns each step into
        # thousands of random small reads on Lustre. 9.04e9 uint16 is 18.1 GB,
        # which fits a shared GPU node's RAM share; TILT_DATA_MMAP=1 is the
        # escape hatch if a future corpus genuinely will not.
        use_mmap = os.environ.get("TILT_DATA_MMAP", "0") == "1"
        kw = {"mmap_mode": "r"} if use_mmap else {}
        stream = np.load(datadir / "stream.npy", **kw)
        eot_pos = np.load(datadir / "eot_pos.npy")
        bits = _load_bits_by_token_id(embed_dir)
        x = StreamCropX(stream, eot_pos, bits, meta["seq_len"],
                        seed=meta.get("crop_seed", 0))
        if x.d != meta["d"]:
            raise ValueError(
                f"meta d={meta['d']} but seq_len={meta['seq_len']} x "
                f"{x.n_bits} bits from {embed_dir} implies d={x.d}")
        if not use_mmap:
            print(f"[tilt.data] loaded stream-backed dataset into RAM: "
                  f"{len(stream):,} tokens ({stream.nbytes/1e9:.2f} GB), "
                  f"{len(eot_pos):,} document boundaries, "
                  f"{x.n:,} candidate offsets", flush=True)
        print(f"[tilt.data] window acceptance {x.acceptance():.2%} "
              f"(legal = no <|endoftext|> inside {meta['seq_len']} tokens)",
              flush=True)
        return x, None, None, meta

    # --- token-backed (large) datasets ------------------------------------
    if (datadir / "tokens.npy").exists():
        meta = json.loads((datadir / "meta.json").read_text())
        embed_dir = Path(meta.get("embed_dir", "text_based_test"))
        if not embed_dir.is_absolute() and not embed_dir.exists():
            embed_dir = datadir.parent.parent / "text_based_test"

        # Load into RAM, do NOT memory-map. BatchSampler draws a fresh set of
        # `batch_size` uniformly random row indices EVERY step, so a memmap
        # turns each step into thousands of random small reads. On a parallel
        # filesystem (e.g. Lustre project storage, which is tuned for
        # large sequential I/O) that is close to a worst case: measured at
        # ~2.1 s/step for batch_size=8192, against ~0.07 s/step once resident.
        # The whole point of the token layout is that it is small -- 30M rows
        # is 630 MB, nothing on a node with hundreds of GB. Set
        # TILT_DATA_MMAP=1 only if a future dataset genuinely will not fit.
        use_mmap = os.environ.get("TILT_DATA_MMAP", "0") == "1"
        kw = {"mmap_mode": "r"} if use_mmap else {}
        tokens = np.load(datadir / "tokens.npy", **kw)
        reward = np.load(datadir / "reward.npy", **kw)
        if not use_mmap:
            nbytes = tokens.nbytes + reward.nbytes
            print(f"[tilt.data] loaded token-backed dataset into RAM: "
                  f"{len(tokens):,} rows, {nbytes/1e9:.2f} GB "
                  f"(set TILT_DATA_MMAP=1 to memory-map instead)", flush=True)
        use_flags = os.environ.get("TILT_TOKEN_FLAGS", "0") == "1"
        bits = _load_bits_by_token_id(embed_dir, flags=use_flags)
        x = TokenBackedX(tokens, reward, bits)
        # meta["d"] records the dataset's OWN d, built from the 15-bit code.
        # Flags widen every token block without changing a single stored token,
        # so the expected d grows by seq_len * n_flags and the check still
        # catches a genuine mismatch rather than being switched off.
        n_flags = bits.shape[1] - 15 if use_flags else 0
        expect = meta["d"] + tokens.shape[1] * n_flags
        if x.d != expect:
            raise ValueError(
                f"meta d={meta['d']} (+{tokens.shape[1] * n_flags} flag coords) "
                f"= {expect}, but tokens imply d={x.d}")
        if n_flags:
            # Make the RETURNED meta describe the data actually returned. Every
            # downstream consumer -- train(), train_selfcond(), probe_xf_v3,
            # run_train_transformer -- validates cfg.d against meta["d"], and
            # patching each one is whack-a-mole that misses the next one. The
            # dict is an in-memory copy of the json, so the file on disk is
            # untouched and the composed width is preserved under d_composed.
            meta["d_composed"] = meta["d"]
            meta["d"] = x.d
            print(f"[tilt.data] loaded view d={x.d} "
                  f"(composed d={meta['d_composed']}, +{n_flags} flag "
                  f"coords/token); dataset on disk unchanged", flush=True)
        if meta.get("freeze_noise"):
            raise NotImplementedError(
                "freeze_noise is not supported for token-backed datasets")
        target = _T.load(target_path) if target_path.exists() else None
        return x, None, target, meta

    # --- image-backed (mnist) datasets ------------------------------------
    # Written by scripts/make_mnist_data.py: images.npy (n, 784) uint8 +
    # labels.npy (n,) uint8, holdout_* siblings for the 10k test split. Same
    # contract as the other compact views: 794 bytes/row, bits materialised
    # for sampled rows only (8-bit Gray, patch-major -- see tilt/mnist.py).
    if (datadir / "images.npy").exists():
        from .mnist import ImageBackedX

        meta = json.loads((datadir / "meta.json").read_text())
        use_mmap = os.environ.get("TILT_DATA_MMAP", "0") == "1"
        kw = {"mmap_mode": "r"} if use_mmap else {}
        images = np.load(datadir / "images.npy", **kw)
        labels = np.load(datadir / "labels.npy", **kw)
        x = ImageBackedX(images, labels)
        if not use_mmap:
            print(f"[tilt.data] loaded image-backed dataset into RAM: "
                  f"{len(images):,} rows, {(images.nbytes + labels.nbytes)/1e6:.0f} MB",
                  flush=True)
        if x.d != meta["d"]:
            raise ValueError(f"meta d={meta['d']} but images imply d={x.d}")
        if meta.get("freeze_noise"):
            raise NotImplementedError(
                "freeze_noise is not supported for image-backed datasets")
        return x, None, None, meta

    # --- grid-backed (large sudoku6) datasets -----------------------------
    # Same contract as TokenBackedX above: 37 bytes/row instead of ~280 of CSV
    # text, bits materialised for sampled rows only. Written by
    # scripts/make_sudoku6_data.py --format npy.
    if (datadir / "grids.npy").exists():
        from .data_grids import GridBackedX

        meta = json.loads((datadir / "meta.json").read_text())
        use_mmap = os.environ.get("TILT_DATA_MMAP", "0") == "1"
        x = GridBackedX.load(datadir, mmap=use_mmap)
        if not use_mmap:
            print(f"[tilt.data] loaded grid-backed dataset into RAM: "
                  f"{len(x):,} rows, {x.nbytes()/1e6:.0f} MB "
                  f"(set TILT_DATA_MMAP=1 to memory-map instead)", flush=True)
        if x.d != meta["d"]:
            raise ValueError(f"meta d={meta['d']} but grids imply d={x.d}")
        if meta.get("freeze_noise"):
            raise NotImplementedError(
                "freeze_noise is not supported for grid-backed datasets")
        target = _T.load(target_path) if target_path.exists() else None
        return x, None, target, meta

    # --- dense CSV datasets (unchanged) -----------------------------------
    df = pd.read_csv(datadir / "train.csv")
    meta = json.loads((datadir / "meta.json").read_text())
    target_path = datadir / "target.json"
    target = _T.load(target_path) if target_path.exists() else None
    d = meta["d"]

    x = torch.tensor(df[[f"x{i}" for i in range(d)]].to_numpy(), dtype=torch.float32)
    frozen = None
    if meta.get("freeze_noise"):
        u = torch.tensor(df[[f"u{i}" for i in range(d)]].to_numpy(), dtype=torch.float32)
        a = torch.tensor(df["a"].to_numpy(), dtype=torch.float32).unsqueeze(1)
        frozen = (u, a)
    return x, frozen, target, meta


class BatchSampler:
    """Yields (x, u, a, beta) batches, re-noising every draw by default.

    lambda_max > 1 activates the Lambda augmentation from the write-up: the
    sample is rescaled coordinate-wise before noising, which shifts the natural
    parameters seen during training toward the reward axis without changing the
    operator being learned (since Lambda preserves the norm on the cube).

    mode1_prob > 0 additionally activates infill-mode focusing (write-up
    Section 8.4): with probability mode1_prob per row, the reward-focusing
    profile (q = ones, "mode 0") is replaced by a sparse random-k-subset
    profile ("mode 1", noising.sample_infill_q) that zeroes out the other
    coordinates instead of stretching a single reward axis -- this is what
    manufactures training points that look like "some coordinates fixed, the
    rest free," the shape an infilling query needs. mode1_prob=0 (default)
    never draws mode 1, so existing lambda-only runs are reproduced exactly.

    beta_focus_prob > 0 (with beta_focus_range set) additionally oversamples
    a narrower "hard" beta sub-range on top of the base log-uniform
    [beta_lo, beta_hi] draw: w.p. beta_focus_prob per row, beta is instead
    drawn log-uniform on beta_focus_range. This targets a beta band found
    empirically harder to fit (e.g. beta~0.5-2.5, where tanh saturation
    hasn't kicked in yet but the signal is no longer dominated by the
    marginal mean) without narrowing beta_lo/beta_hi themselves, so the full
    range is still covered. beta_focus_prob=0 (default) never draws the
    focus band, reproducing existing runs exactly (same RNG draws as before).
    """

    def __init__(self, x: torch.Tensor, beta_lo: float = 0.02, beta_hi: float = 50.0,
                 frozen=None, lambda_max: float = 1.0, mode1_prob: float = 0.0,
                 lambda1_prob: float = 0.25, beta_focus_range: tuple[float, float] | None = None,
                 beta_focus_prob: float = 0.0, seed: int = 0, n_extra: int = 1):
        self.x = x
        self.n, self.d = x.shape
        self.beta_lo, self.beta_hi = beta_lo, beta_hi
        self.frozen = frozen
        self.lambda_max = lambda_max
        self.mode1_prob = mode1_prob
        self.lambda1_prob = lambda1_prob
        self.beta_focus_range = beta_focus_range
        self.beta_focus_prob = beta_focus_prob
        # Coordinates past the token blocks. 1 = the reward axis (every run
        # before owt_v1); 0 = an untilted run with no reward coordinate.
        self.n_extra = n_extra
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, bs: int):
        idx = torch.randint(0, self.n, (bs,), generator=self.g)
        x = self.x[idx]
        if self.frozen is not None:
            u, a = self.frozen[0][idx], self.frozen[1][idx]
            from .noising import beta_of_a

            return x, u, a, beta_of_a(a)

        u, a, beta, diag = _augmented_noise(
            x, self.beta_lo, self.beta_hi, self.lambda_max, self.mode1_prob,
            self.lambda1_prob, self.beta_focus_range, self.beta_focus_prob,
            generator=self.g, n_extra=self.n_extra,
        )
        return x, u, a, beta, diag


def _augmented_noise(x: torch.Tensor, beta_lo: float, beta_hi: float,
                      lambda_max: float, mode1_prob: float, lambda1_prob: float,
                      beta_focus_range: tuple[float, float] | None,
                      beta_focus_prob: float, generator, n_extra: int = 1) -> tuple:
    """Draw beta (with optional focus-band oversampling) and optional Lambda/
    mode1 diag augmentation for a FIXED batch of samples `x`, then VP-noise
    it. This is the shared tail of `BatchSampler.__call__`'s per-batch logic
    (everything after `x` is chosen), factored out so `build_frozen_holdout`
    can reuse the exact same augmentation recipe on a one-time fixed `x`
    (no with-replacement resampling) instead of on a fresh per-call
    minibatch. Contains no additional/reordered RNG calls relative to the
    pre-refactor `BatchSampler.__call__` body, so existing seeded training
    runs reproduce bit-for-bit.

    Returns (u, a, beta, diag) -- see `BatchSampler.__call__`'s trailing
    comment for what `u`/`diag` mean and how to turn them into a network
    query (`noising.build_query`).
    """
    bs, d = x.shape
    beta = sample_beta_loguniform(bs, beta_lo, beta_hi, generator=generator)
    if beta_focus_prob > 0.0 and beta_focus_range is not None:
        flo, fhi = beta_focus_range
        beta_focus = sample_beta_loguniform(bs, flo, fhi, generator=generator)
        focus_mask = torch.rand(bs, 1, generator=generator) < beta_focus_prob
        beta = torch.where(focus_mask, beta_focus, beta)
    a = a_of_beta(beta)
    xs = x
    diag = None
    if lambda_max > 1.0 or mode1_prob > 0.0:
        # core = the coordinates that are NOT the reward axis. n_extra=1 is the
        # historical geometry (d-1 token bits plus x_*); n_extra=0 is an untilted
        # run with no reward coordinate at all, where every coordinate is core
        # and there is no lambda axis to stretch. Hardcoding `d - 1` here silently
        # left one coordinate un-profiled at n_extra=0 and made profile_diag
        # return a (bs, d+1) diagonal against a (bs, d) sample.
        if n_extra == 0 and lambda_max > 1.0:
            raise ValueError(
                f"lambda_max={lambda_max} > 1 with n_extra=0: there is no reward "
                f"axis to stretch. Set lambda_max=1.0 for untilted runs.")
        core = d - n_extra
        # Mode: 0 (reward-focusing, q = ones) w.p. 1 - mode1_prob, 1
        # (infill-focusing, sparse random-k-subset q) w.p. mode1_prob.
        q_ones = torch.ones(bs, core)
        if mode1_prob > 0.0:
            q_infill = sample_infill_q(bs, core, generator=generator)
            mode1 = torch.rand(bs, 1, generator=generator) < mode1_prob
            q = torch.where(mode1, q_infill, q_ones)
        else:
            q = q_ones
        # lambda: exactly 1 (no reward-axis stretch) w.p. lambda1_prob when
        # augmentation is otherwise active, else log-uniform on [1, lambda_max].
        if lambda_max > 1.0:
            u01 = torch.rand(bs, 1, generator=generator)
            lam_loguniform = lambda_max ** u01
            lam1 = torch.rand(bs, 1, generator=generator) < lambda1_prob
            lam = torch.where(lam1, torch.ones(bs, 1), lam_loguniform)
        else:
            lam = torch.ones(bs, 1)
        # With n_extra=0 the budget identity is already satisfied by q alone:
        # sample_infill_q guarantees sum(q**2) == core == d, and profile_diag's
        # scale factor is exactly 1 at lam == 1, so diag = q is the correct
        # (and only admissible) Lambda.
        diag = q if n_extra == 0 else profile_diag(lam, q, n_extra)
        xs = diag * x
    # u is the RAW noisy observation -- literally what the noising process produces,
    # U = a * Lambda * X + sqrt(1-a^2) Z.  diag is returned alongside it (None when no
    # Lambda/infill augmentation) so the caller can build the correct query point for G:
    # since ||Lambda x||^2 = d is constant on the cube, p(x|u) ~ pi(x) exp(beta(a) <Lambda
    # u, x>), i.e. the natural parameter is beta(a) * Lambda @ u, NOT beta(a) * u.  Building
    # that query (and renormalising it) is the caller's job -- see noising.build_query.
    u = vp_noise(xs, a, generator=generator)
    return u, a, beta, diag


def build_frozen_holdout(x_sampler, n: int, beta_lo: float = 0.02, beta_hi: float = 50.0,
                          lambda_max: float = 1.0, mode1_prob: float = 0.0,
                          lambda1_prob: float = 0.25,
                          beta_focus_range: tuple[float, float] | None = None,
                          beta_focus_prob: float = 0.0, seed: int = 0,
                          n_extra: int = 1) -> dict:
    """Build a FIXED (x, u, a, beta, diag) holdout batch for ground-truth-free
    test-loss tracking during training (see `train()`'s `holdout` argument in
    tilt/train.py and tilt/train_selfcond.py).

    `x_sampler(n, generator) -> (n, d)` must draw FRESH samples from the
    target's own generative process -- e.g. a small lambda wrapping
    `tilt.sudoku6.sample_weak_rows` plus the reward coordinate for sudoku6.
    Critically, this must be a source disjoint from whatever finite row pool
    `BatchSampler` draws its own training `x`'s from (e.g. NOT the same
    `train.csv` `BatchSampler` indexes with replacement every batch) --
    otherwise "holdout" loss would just be in-sample loss on repeated rows,
    exactly the situation this function exists to avoid.

    Draws beta (with optional focus-band oversampling) and Lambda/mode1 diag
    augmentation via the same recipe (`_augmented_noise`, shared with
    `BatchSampler`) a training run actually uses, so the resulting regression
    loss `E||X - G_theta(v,a*)||^2` (v,a* from `noising.build_query`) is
    directly comparable to that run's own training loss -- no exact
    ground-truth operator is needed at all, unlike `evaluate.py`'s
    `operator_error`/`sse_eval_fn`, which is what makes this usable at any d
    (e.g. the 150-bit `rhyme_gpt2_v1` text domain, where no exact operator
    exists). The draw happens ONCE (not resampled with replacement like
    `BatchSampler`), so all `n` points are distinct and the returned tuple is
    meant to be reused unchanged at every eval checkpoint of a run (and
    across separate runs/checkpoints being compared against each other, via
    `save_frozen_holdout`/`load_frozen_holdout`).

    Returns {"x", "u", "a", "beta", "diag"} (diag is None if no Lambda/mode1
    augmentation was requested).
    """
    g = torch.Generator().manual_seed(seed)
    x = x_sampler(n, g)
    u, a, beta, diag = _augmented_noise(
        x, beta_lo, beta_hi, lambda_max, mode1_prob, lambda1_prob,
        beta_focus_range, beta_focus_prob, generator=g, n_extra=n_extra,
    )
    return {"x": x, "u": u, "a": a, "beta": beta, "diag": diag}


def save_frozen_holdout(holdout: dict, path: str | Path) -> None:
    """Persist a `build_frozen_holdout` dict to disk (plain `torch.save`) so the
    identical holdout batch can be reused across separate training runs/
    processes being compared against each other."""
    torch.save(holdout, path)


def load_frozen_holdout(path: str | Path) -> dict:
    """Load a holdout dict saved by `save_frozen_holdout`."""
    return torch.load(path)


def holdout_loss(net, holdout: dict, d: int, self_cond: bool = False) -> float:
    """Ground-truth-free regression loss E||X - G_theta(v,a*)||^2 of `net` on
    a frozen holdout batch (see `build_frozen_holdout`) -- the exact same
    quantity the training loop itself minimises, evaluated on data that
    `BatchSampler` never draws its training `x`'s from.

    `self_cond=True` scores `net.self_cond_predict` (the deployed two-pass
    estimate for a `SelfCondTiltMLP`) instead of a plain 2-arg forward call --
    mirrors why `evaluate_selfcond.py` exists at all: a plain forward call
    would silently default `x_pred` to zero and understate a self-conditioned
    network's real accuracy.
    """
    from .noising import build_query

    v, a_star = build_query(holdout["u"], holdout["a"], holdout["diag"], d=d)
    with torch.no_grad():
        pred = net.self_cond_predict(v, a_star) if self_cond else net(v, a_star)
        return float((holdout["x"] - pred).pow(2).sum(1).mean())
