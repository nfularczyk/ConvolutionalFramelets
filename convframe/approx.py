"""Compressed reconstructions for the numerical examples (Sec. 4.2, Figs. 7-8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .framelets import Framelet2D
from .hankel import hankel_matrix, structure_add_1d
from .metrics import compression_ratio, hosvd_storage, psnr, relative_error, svd_storage

Array = NDArray[np.floating]


@dataclass
class Approximation:
    image: Array
    relative_error: float
    psnr: float
    compression_ratio: float
    n_uncompressed: int
    n_stored: int
    label: str


def tensor_hosvd_approx(
    f: Array,
    patch: int | Tuple[int, int],
    ranks: Sequence[int],
    data_range: float | None = None,
) -> Approximation:
    """Bivariate convolutional framelets (Algorithm 2).

    Truncated HOSVD of the patch tensor with multilinear rank
    ``(r_0, r_1, r_2, r_3)``, then structure-add (eq. (6)).
    """
    f = np.asarray(f, dtype=float)
    n1, n2 = f.shape
    if isinstance(patch, tuple):
        ell1, ell2 = patch
    else:
        ell1 = ell2 = int(patch)
    model, core = Framelet2D.from_hosvd(f, (ell1, ell2), ranks)
    rec = model.reconstruct_from(core)
    n_full = n1 * n2 * ell1 * ell2
    n_store = hosvd_storage(ranks, (n1, n2), (ell1, ell2))
    return Approximation(
        image=rec,
        relative_error=relative_error(f, rec),
        psnr=psnr(f, rec, data_range=data_range),
        compression_ratio=compression_ratio(n_full, n_store),
        n_uncompressed=n_full,
        n_stored=n_store,
        label=f"patch tensor HOSVD ranks {tuple(int(r) for r in ranks)}",
    )


def _square_window(patch: int | Tuple[int, int], name: str) -> int:
    if isinstance(patch, tuple):
        if len(patch) != 2 or patch[0] != patch[1]:
            raise ValueError(f"{name} needs a square patch, got {patch}")
        return int(patch[0])
    return int(patch)


def vectorized_hankel_svd_approx(
    f: Array,
    patch: int | Tuple[int, int],
    rank: int,
    data_range: float | None = None,
) -> Approximation:
    """Univariate convolutional framelets on the vectorized image (Algorithm 1).

    Flatten the ``N x N`` image, form the ``(N^2, ell)`` periodic Hankel
    matrix H, truncate its SVD to rank r, structure-add (eq. (2)), reshape.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {f.shape}")
    n1, n2 = f.shape
    ell = _square_window(patch, "vectorized_hankel_svd_approx")
    vec = f.reshape(-1)
    n = vec.shape[0]
    if not 1 <= ell <= n:
        raise ValueError(f"need 1 <= ell <= N^2, got ell={ell}, N^2={n}")
    H = hankel_matrix(vec, ell)
    n_rows, n_cols = H.shape
    if not 1 <= rank <= min(n_rows, n_cols):
        raise ValueError(
            f"rank={rank} is outside 1..{min(n_rows, n_cols)} for a "
            f"{n_rows} x {n_cols} Hankel matrix"
        )
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    H_hat = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
    rec = structure_add_1d(H_hat).reshape(n1, n2)
    n_full = n_rows * n_cols
    n_store = svd_storage(n_rows, n_cols, rank)
    return Approximation(
        image=rec,
        relative_error=relative_error(f, rec),
        psnr=psnr(f, rec, data_range=data_range),
        compression_ratio=compression_ratio(n_full, n_store),
        n_uncompressed=n_full,
        n_stored=n_store,
        label=f"patch matrix SVD rank {rank}",
    )
