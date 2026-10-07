"""MNIST on the hypercube: 8-bit Gray-coded pixels + a 10-bit one-hot class block.

GEOMETRY
--------
    28x28 pixels x 8 Gray bits          = 6272 core coordinates
    10 one-hot class bits (+1/-1)       =   10 reward coordinates
                                     d  = 6282,  {-1,+1}^d

Gray code, not plain binary: adjacent intensities differ by exactly one bit,
so Hamming distance tracks intensity locally (plain binary flips all 8 bits
at 127->128). The encoding is a per-pixel bijection, so nothing downstream
(operator, sampler, tilts) knows or cares -- only encode/decode here do.

COORDINATE ORDER IS PATCH-MAJOR, matched to the transformer featuriser.
TransformerTilt slices u[:, :K*M].view(B, K, M), so coordinates must come
grouped by token. Token k is the k-th 4x4 patch (7x7 grid of patches,
row-major), within a patch pixels are row-major, within a pixel the 8 Gray
bits are MSB-first: K=49 tokens x M=128 bits. The reward block occupies
coords 6272..6281, digit c one-hot: +1 at coordinate 6272+c, -1 elsewhere.

CLASS CONDITIONING IS A PATTERN TILT, NOT AN AXIS TILT. The class-c reward
pattern r_c has ALL TEN coordinates at +/-1, and block-Lambda training
stretches all ten together -- so the sampler must tilt along the full
pattern, h_extra = tau * r_c, not tau * e_c. (Separation between classes is
e^{4 tau} per unit tau, vs e^{2 tau} for an axis tilt: tau_pattern =
tau_axis / 2 when comparing dials.) The t=0 query then sits at the singular
endpoint of the block-Lambda family, per-coordinate magnitude
sqrt(d / n_extra) -- which is why LAMBDA_MAX admissibility is
sqrt(d / 10) = 25.06 here, NOT sqrt(d) = 79.3. See noising.profile_diag.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

IMG = 28
PATCH = 4
N_PATCH = (IMG // PATCH) ** 2          # 49 tokens
BITS = 8                               # Gray bits per pixel
TOKEN_BITS = PATCH * PATCH * BITS      # 128 bits per token
N_CLASSES = 10
CORE = IMG * IMG * BITS                # 6272
D = CORE + N_CLASSES                   # 6282

# Admissible block stretch is lam < sqrt(d / n_extra) (see profile_diag).
# 24.0 is 0.958 of the 25.064 ceiling -- the same "train close beneath the
# endpoint" fraction the sudoku6 -> sudoku9 transfer used against ITS ceiling.
LAMBDA_MAX_CEILING = float(np.sqrt(D / N_CLASSES))
LAMBDA_MAX_RECOMMENDED = 24.0

# ---------------------------------------------------------------------------
# Gray code lookup tables
# ---------------------------------------------------------------------------
_v = np.arange(256, dtype=np.uint8)
GRAY_ENC = (_v ^ (_v >> 1)).astype(np.uint8)          # value -> gray byte
GRAY_DEC = np.zeros(256, dtype=np.uint8)              # gray byte -> value
GRAY_DEC[GRAY_ENC] = _v
# gray byte -> 8 bits, MSB first, {0,1}
GRAY_BITS = ((GRAY_ENC[:, None] >> np.arange(7, -1, -1)[None, :]) & 1).astype(np.uint8)

# ---------------------------------------------------------------------------
# Patch-major pixel permutation
# ---------------------------------------------------------------------------
def _patch_perm() -> np.ndarray:
    """perm[k] = flat (row-major) pixel index of the k-th patch-major pixel."""
    idx = np.arange(IMG * IMG).reshape(IMG, IMG)
    out = []
    for pr in range(IMG // PATCH):
        for pc in range(IMG // PATCH):
            out.append(idx[pr * PATCH:(pr + 1) * PATCH,
                           pc * PATCH:(pc + 1) * PATCH].ravel())
    return np.concatenate(out)

PATCH_PERM = _patch_perm()                       # patch-major -> row-major
PATCH_PERM_INV = np.argsort(PATCH_PERM)          # row-major -> patch-major

# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------
def encode_image_batch(imgs: np.ndarray, labels: np.ndarray) -> torch.Tensor:
    """(b, 784) or (b, 28, 28) uint8 intensities + (b,) labels -> (b, d) float32 in {-1,+1}."""
    imgs = np.asarray(imgs, dtype=np.uint8).reshape(len(imgs), IMG * IMG)
    # intp, NOT the caller's dtype: labels.npy is uint8 and `CORE + labels`
    # would then be computed in uint8, where 6272 does not fit (numpy 2 raises
    # OverflowError rather than wrapping silently).
    labels = np.asarray(labels, dtype=np.intp)
    b = len(imgs)
    out = np.empty((b, D), dtype=np.float32)
    bits = GRAY_BITS[imgs[:, PATCH_PERM]]                 # (b, 784, 8) {0,1}
    out[:, :CORE] = bits.reshape(b, CORE).astype(np.float32) * 2.0 - 1.0
    out[:, CORE:] = -1.0
    out[np.arange(b), CORE + labels] = 1.0
    return torch.from_numpy(out)


def decode_bits(x: torch.Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(b, d) real-valued (signs are taken) -> ((b, 28, 28) uint8, (b,) labels).

    Labels are read off the reward block by argmax (works for one-hot and for
    anything a sampler produced, however non-one-hot).
    """
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    b = len(x)
    bits = (x[:, :CORE] > 0).astype(np.uint8).reshape(b, IMG * IMG, BITS)
    gray = np.zeros((b, IMG * IMG), dtype=np.uint8)
    for j in range(BITS):                                  # MSB first
        gray = (gray << 1) | bits[:, :, j]
    vals = GRAY_DEC[gray]                                  # patch-major order
    imgs = vals[:, PATCH_PERM_INV].reshape(b, IMG, IMG)
    labels = x[:, CORE:].argmax(axis=1)
    return imgs, labels


# ---------------------------------------------------------------------------
# class patterns and tilts
# ---------------------------------------------------------------------------
def class_pattern(c: int) -> torch.Tensor:
    """(10,) float32 one-hot pattern r_c in {-1,+1}: +1 at c, -1 elsewhere."""
    r = -torch.ones(N_CLASSES)
    r[c] = 1.0
    return r


def reward_tilt(tau: float, c: int) -> torch.Tensor:
    """(d,) additive natural parameter h_extra = tau * r_c on the reward block.

    THE PATTERN TILT -- all ten reward coordinates, not just e_c. This is the
    direction block-Lambda training saturates; tilting a single axis queries
    early-trajectory geometry the training measure never manufactures.
    Feed to noising.build_query / the sampler as h_extra.
    """
    h = torch.zeros(D)
    h[CORE:] = tau * class_pattern(c)
    return h


# ---------------------------------------------------------------------------
# dataset view (BatchSampler contract: .shape + index __getitem__)
# ---------------------------------------------------------------------------
class ImageBackedX:
    """Lazy (n, d) float32 view backed by (n, 784) uint8 images + (n,) labels.

    Same contract as TokenBackedX / GridBackedX: BatchSampler only uses
    `.shape` and `__getitem__` with an index tensor; bits are materialised
    for the sampled rows only. 794 bytes/row vs 25 KB dense float32.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        if images.ndim != 2 or images.shape[1] != IMG * IMG:
            raise ValueError(f"expected images (n, {IMG * IMG}), got {images.shape}")
        if len(labels) != len(images):
            raise ValueError(f"{len(images)} images but {len(labels)} labels")
        if int(labels.max()) >= N_CLASSES:
            raise ValueError(f"label {labels.max()} out of range")
        self.images = images
        self.labels = labels
        self.n = int(images.shape[0])
        self.d = D
        self.shape = (self.n, self.d)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx) -> torch.Tensor:
        i = idx.cpu().numpy() if torch.is_tensor(idx) else np.asarray(idx)
        return encode_image_batch(np.asarray(self.images[i]),
                                  np.asarray(self.labels[i]))

    def dense(self) -> torch.Tensor:
        """Materialise everything -- (60k, 6282) float32 is 1.5 GB; fine once,
        not per step."""
        return self[np.arange(self.n)]


if __name__ == "__main__":  # python -m tilt.mnist
    rng = np.random.default_rng(0)

    # Gray tables: bijection + adjacency
    assert len(set(GRAY_ENC.tolist())) == 256
    assert (GRAY_DEC[GRAY_ENC] == _v).all()
    adj = GRAY_ENC[:-1] ^ GRAY_ENC[1:]
    assert all(bin(int(x)).count("1") == 1 for x in adj), "Gray adjacency broken"

    # permutation really is one
    assert sorted(PATCH_PERM.tolist()) == list(range(784))
    assert (PATCH_PERM[PATCH_PERM_INV] == np.arange(784)).all()

    # encode/decode roundtrip
    imgs = rng.integers(0, 256, size=(64, IMG, IMG), dtype=np.uint8)
    # uint8 ON PURPOSE -- this is what labels.npy holds, and an int64 label
    # array hides the CORE + labels overflow that shipped once already.
    labels = rng.integers(0, 10, size=64).astype(np.uint8)
    x = encode_image_batch(imgs, labels)
    assert x.shape == (64, D) and set(x.unique().tolist()) == {-1.0, 1.0}
    imgs2, labels2 = decode_bits(x)
    assert (imgs2 == imgs).all() and (labels2 == labels).all()

    # reward block is one-hot
    assert (x[:, CORE:].sum(1) == -(N_CLASSES - 2)).all()

    # pattern tilt separation: <r_c, r_c> - <r_c, r_c'> == 4
    r0, r1 = class_pattern(0), class_pattern(1)
    assert float(r0 @ r0 - r0 @ r1) == 4.0

    # budget identity + n_extra=1 backward compat of profile_diag
    from .noising import profile_diag
    lam = torch.tensor([[1.0], [5.0], [24.0]])
    q = torch.ones(3, CORE)
    diag = profile_diag(lam, q, n_extra=N_CLASSES)
    assert diag.shape == (3, D)
    assert torch.allclose((diag ** 2).sum(1), torch.full((3,), float(D)), atol=1e-3)
    # old single-axis formula reproduced exactly at n_extra=1
    q1 = torch.ones(3, 150)
    old = torch.cat([torch.sqrt((1.0 + (1.0 - lam**2) / 150).clamp_min(0)) * q1, lam], -1)
    assert torch.equal(profile_diag(lam, q1), old) and torch.equal(
        profile_diag(lam, q1, n_extra=1), old)
    assert abs(LAMBDA_MAX_CEILING - 25.0639) < 1e-3

    # ImageBackedX matches direct encoding
    xv = ImageBackedX(imgs.reshape(64, -1), labels)
    idx = torch.tensor([3, 1, 4, 1, 5])
    assert torch.equal(xv[idx], encode_image_batch(imgs.reshape(64, -1)[idx.numpy()],
                                                   labels[idx.numpy()]))
    print(f"tilt.mnist OK: d={D} ({N_PATCH}x{TOKEN_BITS} + {N_CLASSES}), "
          f"lambda ceiling sqrt(d/{N_CLASSES})={LAMBDA_MAX_CEILING:.4f}, "
          f"recommended {LAMBDA_MAX_RECOMMENDED}")
