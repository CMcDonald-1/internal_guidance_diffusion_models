"""Build the mnist_v1 dataset: raw IDX files -> images.npy/labels.npy + meta.json.

What gets stored is the compact form -- (n, 784) uint8 intensities + (n,)
uint8 labels, 785 bytes/row -- NOT bits and NOT noised pairs, per the same
design note as every other dataset here: samples are the expensive artifact,
Gray-encoding is a deterministic view (tilt/mnist.py), and noise is free.

The 10k official test split is stored beside the training set as
holdout_images.npy / holdout_labels.npy. It is the frozen-holdout source for
run_train_mnist_reward.py -- the direct memorisation watch for a 60k-image
pool being resampled ~1k+ epochs -- and later the memcheck reference.

Sources, in order of preference:
    --raw DIR   directory already containing the four IDX files (gz or not):
                train-images-idx3-ubyte.gz  train-labels-idx1-ubyte.gz
                t10k-images-idx3-ubyte.gz   t10k-labels-idx1-ubyte.gz
    (else)      downloaded from --mirror (default: the ossci S3 mirror --
                yann.lecun.com 403s scripted pulls these days). On a cluster
                whose compute nodes have no internet, run this on a login node.

Usage:
    python scripts/make_mnist_data.py --out data/mnist_v1
    python scripts/make_mnist_data.py --out data/mnist_v1 --raw data/mnist_raw
"""
from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tilt.mnist import D, N_CLASSES, N_PATCH, TOKEN_BITS

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
DEFAULT_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def _read_idx(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        if magic >> 8 != 0x08 or ndim not in (1, 3):
            raise ValueError(f"{path}: unexpected IDX magic {magic:#x}")
        dims = struct.unpack(f">{ndim}I", f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        if data.size != int(np.prod(dims)):
            raise ValueError(f"{path}: {data.size} bytes for dims {dims}")
        return data.reshape(dims)


def _download(url: str, dest: Path) -> None:
    """urllib first, then the system curl/wget.

    Some cluster login nodes run a uv-managed CPython whose SSL context has no CA
    bundle, so urllib dies with CERTIFICATE_VERIFY_FAILED while the system
    curl (which has its own bundle) works fine. Try both before giving up,
    and if both fail say what to actually do about it.
    """
    try:
        urllib.request.urlretrieve(url, dest)
        return
    except Exception as exc:  # noqa: BLE001 -- retry on anything
        print(f"  urllib failed ({type(exc).__name__}); trying curl/wget",
              flush=True)
    import shutil
    import subprocess

    for cmd in (["curl", "-fsSL", "-o", str(dest), url],
                ["wget", "-q", "-O", str(dest), url]):
        if shutil.which(cmd[0]) and subprocess.run(cmd).returncode == 0:
            return
    dest.unlink(missing_ok=True)
    raise SystemExit(
        f"could not download {url}.\n"
        f"  This is usually a missing CA bundle (some cluster login nodes: a\n"
        f"  uv-managed python has none, though the system curl does).\n"
        f"  Options, in order of preference:\n"
        f"    1. Build the dataset on a machine with working networking and\n"
        f"       scp the four .npy files + meta.json across -- 55 MB, and it\n"
        f"       guarantees byte-identical data on both sides.\n"
        f"    2. Fetch the four IDX files by hand into a directory and pass\n"
        f"       --raw <that directory>.\n"
        f"    3. export SSL_CERT_FILE=$(python -c 'import certifi;"
        f"print(certifi.where())') and re-run.")


def _locate(raw: Path | None, name: str, mirror: str, cache: Path) -> Path:
    fname = FILES[name]
    if raw is not None:
        for cand in (raw / fname, raw / fname[:-3]):  # .gz or already unzipped
            if cand.exists():
                return cand
        raise SystemExit(f"--raw {raw} does not contain {fname} (or unzipped)")
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / fname
    if not dest.exists():
        url = mirror.rstrip("/") + "/" + fname
        print(f"  downloading {url}", flush=True)
        _download(url, dest)
    return dest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/mnist_v1")
    p.add_argument("--raw", default=None,
                   help="directory with the four IDX files; skips download")
    p.add_argument("--mirror", default=DEFAULT_MIRROR)
    p.add_argument("--cache", default="data/mnist_raw",
                   help="where downloads land (kept for reproducibility)")
    p.add_argument("--beta-lo", type=float, default=0.02)
    p.add_argument("--beta-hi", type=float, default=50.0)
    a = p.parse_args()

    raw = Path(a.raw) if a.raw else None
    paths = {k: _locate(raw, k, a.mirror, Path(a.cache)) for k in FILES}

    tri = _read_idx(paths["train_images"])
    trl = _read_idx(paths["train_labels"])
    tei = _read_idx(paths["test_images"])
    tel = _read_idx(paths["test_labels"])
    if tri.shape != (60000, 28, 28) or tei.shape != (10000, 28, 28):
        raise SystemExit(f"unexpected shapes: train {tri.shape}, test {tei.shape}")
    if len(trl) != 60000 or len(tel) != 10000:
        raise SystemExit("label counts wrong")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "images.npy", tri.reshape(60000, 784))
    np.save(out / "labels.npy", trl.astype(np.uint8))
    np.save(out / "holdout_images.npy", tei.reshape(10000, 784))
    np.save(out / "holdout_labels.npy", tel.astype(np.uint8))

    counts = np.bincount(trl, minlength=N_CLASSES).tolist()
    meta = {
        "kind": "mnist", "source": "mnist-idx",
        "n": 60000, "holdout_n": 10000, "d": D,
        "seq_len": N_PATCH, "token_bits": TOKEN_BITS, "n_extra": N_CLASSES,
        "encoding": "gray8_patch4_major",
        "beta_lo": a.beta_lo, "beta_hi": a.beta_hi,
        "class_counts": counts,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}: 60,000 train + 10,000 holdout, d={D} "
          f"({N_PATCH}x{TOKEN_BITS} + {N_CLASSES} class bits)")
    print(f"class counts: {counts}")


if __name__ == "__main__":
    main()
