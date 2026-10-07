"""Compact storage for sudoku datasets: store grids, materialise bits on demand.

Same idea as `data.TokenBackedX`, for a different domain. A sudoku row is a
block of core bits plus a reward coordinate, but those bits are a deterministic
function of the cell values. So the compact form is

    grids.npy   (n, cells) uint8   values 1..N_DIGITS
    reward.npy  (n,)       int8    +1 / -1

    6x6:  36 cells x 3 bits + reward = d 109,   37 bytes/row vs ~280 of CSV
      5M rows :  185 MB   vs  ~1.4 GB CSV
     20M rows :  740 MB   vs  ~5.5 GB CSV

    9x9:  81 cells x 4 bits + reward = d 325,   82 bytes/row vs ~840 of CSV
     10M rows :  820 MB   vs  ~8.4 GB CSV

The board size is inferred from `grids.shape[1]` -- 36 dispatches to
`tilt.sudoku6`, 81 to `tilt.sudoku9`. Nothing else changes, and the 6x6 path is
byte-identical to before.

`BatchSampler` only ever uses `.shape` and `__getitem__` with an index tensor,
so exposing those two is enough for train(), the sampler, and every downstream
caller to work unmodified -- exactly the contract TokenBackedX established.

Bits are materialised for the SAMPLED ROWS ONLY, via the same
`encode_grid_batch` used to build the data, so the layout is identical to what a
CSV dataset would have produced: the leading coords are cell bits (row-major,
MSB first) and the final coord is the reward.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# cells -> (rows, cols, bits_per_cell, encoder module)
_BOARDS = {
    36: (6, 6, 3, "sudoku6"),
    81: (9, 9, 4, "sudoku9"),
}


class GridBackedX:
    """Lazy (n, d) float32 view backed by (n, cells) uint8 grids + (n,) rewards."""

    def __init__(self, grids: np.ndarray, reward: np.ndarray):
        if grids.ndim != 2 or grids.shape[1] not in _BOARDS:
            raise ValueError(
                f"expected grids (n, cells) with cells in {sorted(_BOARDS)}, "
                f"got {grids.shape}")
        if len(reward) != len(grids):
            raise ValueError(f"{len(grids)} grids but {len(reward)} rewards")
        self.grids = grids
        self.reward = reward
        self.n = int(grids.shape[0])
        self.cells = int(grids.shape[1])
        self.rows, self.cols, self.bits_per_cell, self._module = _BOARDS[self.cells]
        self.d = self.cells * self.bits_per_cell + 1
        self.shape = (self.n, self.d)

    def _encoder(self):
        from importlib import import_module

        return import_module(f".{self._module}", __package__).encode_grid_batch

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx) -> torch.Tensor:
        """Row indices (the BatchSampler path), a slice, or a (rows, cols) tuple.

        BatchSampler only ever passes an index tensor, but callers written
        against the dense CSV form use `x[:, :108]`-style 2-D slices, so both
        are supported. NOTE: a full-column slice materialises every row --
        (5M, 109) float32 is 2.2 GB, and (10M, 325) is 13 GB. Prefer
        `valid_grids()` / `.grids` when you want the compact form; that is the
        whole point of this class.
        """
        encode_grid_batch = self._encoder()

        if isinstance(idx, tuple):                       # x[rows, cols]
            rows, cols = idx
            return self[rows][:, cols]
        if isinstance(idx, slice):
            idx = np.arange(*idx.indices(self.n))
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy()
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        elif not np.issubdtype(idx.dtype, np.integer):
            idx = idx.astype(np.int64)
        g = torch.from_numpy(self.grids[idx].astype(np.int64)).reshape(
            -1, self.rows, self.cols)
        core = encode_grid_batch(g)                     # (m, d-1) in {-1,1}
        r = torch.from_numpy(self.reward[idx].astype(np.float32)).reshape(-1, 1)
        return torch.cat([core, r], dim=1)

    def valid_grids(self) -> np.ndarray:
        """(n_valid, cells) uint8 -- the valid grids, straight from the compact form.

        No bit materialisation, no decode round-trip. This is what a
        memorisation check wants: at 10M rows it reads 500k x 81 bytes rather
        than building a 13 GB dense tensor and decoding it back.
        """
        return self.grids[self.reward > 0]

    # -- convenience ---------------------------------------------------------
    def nbytes(self) -> int:
        return int(self.grids.nbytes + self.reward.nbytes)

    @staticmethod
    def exists(datadir: str | Path) -> bool:
        return (Path(datadir) / "grids.npy").exists()

    @classmethod
    def load(cls, datadir: str | Path, mmap: bool = False) -> "GridBackedX":
        """Load into RAM by default.

        BatchSampler draws a fresh set of uniformly random row indices every
        step, so memory-mapping turns each step into thousands of random small
        reads -- pathological on Lustre, per the note in data.TokenBackedX.
        At 37 bytes/row the whole point is that it fits.
        """
        datadir = Path(datadir)
        kw = {"mmap_mode": "r"} if mmap else {}
        grids = np.load(datadir / "grids.npy", **kw)
        reward = np.load(datadir / "reward.npy", **kw)
        return cls(grids, reward)
