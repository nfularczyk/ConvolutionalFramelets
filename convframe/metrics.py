"""Error and storage metrics from Sec. 4.2.

Compression ratio (CR), eq. (19)
    (# entries in the uncompressed patch tensor H or Hankel matrix H)
    / (# entries in the core plus the factor matrices)

Relative error (RE)
    ||f - f_hat||_F  /  ||f||_F

PSNR is the standard peak-signal-to-noise ratio against the given peak.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


def relative_error(original: Array, reconstruction: Array) -> float:
    original = np.asarray(original, dtype=float)
    reconstruction = np.asarray(reconstruction, dtype=float)
    denom = np.linalg.norm(original)
    if denom == 0.0:
        return float(np.linalg.norm(reconstruction))
    return float(np.linalg.norm(original - reconstruction) / denom)


def psnr(original: Array, reconstruction: Array, data_range: float | None = None) -> float:
    original = np.asarray(original, dtype=float)
    reconstruction = np.asarray(reconstruction, dtype=float)
    mse = float(np.mean((original - reconstruction) ** 2))
    if mse == 0.0:
        return float("inf")
    peak = float(np.max(original) - np.min(original)) if data_range is None else float(data_range)
    if peak == 0.0:
        peak = 1.0
    return float(10.0 * np.log10((peak * peak) / mse))


def n_params(*arrays: Array) -> int:
    return int(sum(np.asarray(a).size for a in arrays))


def compression_ratio(uncompressed: int, stored: int) -> float:
    if stored <= 0:
        raise ValueError("stored parameter count must be positive")
    return float(uncompressed) / float(stored)


def hosvd_storage(ranks: Sequence[int], spatial: Sequence[int], patch: Sequence[int]) -> int:
    """Scalars in a truncated HOSVD of the patch tensor.

    ``ranks = (r_0, r_1, r_2, r_3)``: r_0 r_1 r_2 r_3 + N1 r_0 + N2 r_1 + ell1 r_2 + ell2 r_3.
    """
    r0, r1, r2, r3 = (int(r) for r in ranks)
    n1, n2 = (int(s) for s in spatial)
    ell1, ell2 = (int(p) for p in patch)
    core = r0 * r1 * r2 * r3
    factors = n1 * r0 + n2 * r1 + ell1 * r2 + ell2 * r3
    return core + factors


def svd_storage(n_rows: int, n_cols: int, rank: int) -> int:
    """Number of scalars in a rank-r SVD, counting U, singular values, and V."""
    return int(n_rows) * int(rank) + int(rank) + int(n_cols) * int(rank)
