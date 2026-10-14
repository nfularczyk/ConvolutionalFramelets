"""Patch (Hankel) lifting and structure-add inversion for periodic signals.

Notation follows the manuscript (Sec. 2.1 and Sec. 2.3):

    1D patch matrix   F, Hankel(x, ell)          (Fig. 1)
    1D structure-add  f_n = (1/ell) sum_k F[n-1-k, k]             eq. (2)
    2D patch tensor   F[m, n, i, j] = f[m+i, n+j]                 eq. (5)
    2D structure-add  f[n, m] = (1/ell^2) sum_{i,j} F[n-i, m-j, i, j]  eq. (6)

All indices are 0-based and taken mod N.  The patch matrix/tensor is
written F in Secs. 2-3 and H in Sec. 4 (Algorithms 1-2).  Structure-add
(StructureAdd_1 / StructureAdd_2 in Algorithms 1-2) is the left inverse
of the lift: every sample appears in exactly ell (1D) or ell^2 (2D)
patches, so averaging those copies recovers f exactly.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.typing import NDArray

Array = NDArray[np.floating]


def _as_patch(patch: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(patch, tuple):
        if len(patch) != 2:
            raise ValueError("patch must be an int or a pair (ell1, ell2)")
        return int(patch[0]), int(patch[1])
    ell = int(patch)
    return ell, ell


def hankel_matrix(f: Array, ell: int) -> Array:
    """Periodic 1D patch (Hankel) matrix, Hankel(x, ell) in Algorithm 1.

    Returns F of shape (N, ell) with F[m, p] = f[(m + p) mod N], i.e. rows
    start at patch F_0.  The manuscript (Sec. 2.1) starts at F_1,
    F[m, p] = f[m + 1 + p]; the two differ by a circular shift of rows.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 1:
        raise ValueError(f"expected a 1D signal, got shape {f.shape}")
    n = f.shape[0]
    if not 1 <= ell <= n:
        raise ValueError(f"need 1 <= ell <= N, got ell={ell}, N={n}")
    padded = np.concatenate([f, f[: ell - 1]]) if ell > 1 else f
    item = padded.strides[0]
    view = as_strided(padded, shape=(n, ell), strides=(item, item))
    return np.ascontiguousarray(view)


def structure_add_1d(F: Array) -> Array:
    """Structure-add, StructureAdd_1: left inverse of :func:`hankel_matrix`.

    f[n] = (1 / ell) sum_{k=0}^{ell-1} F[(n - k) mod N, k]

    This is eq. (2) with the 0-based row convention of :func:`hankel_matrix`.
    """
    F = np.asarray(F)
    if F.ndim != 2:
        raise ValueError(f"expected a (N, ell) Hankel matrix, got {F.shape}")
    n, ell = F.shape
    out = np.zeros(n, dtype=F.dtype)
    for k in range(ell):
        out += np.roll(F[:, k], k)
    return out / ell


def patch_tensor(f: Array, patch: int | Tuple[int, int] = 8) -> Array:
    """Periodic 2D patch tensor of an image, eq. (5) (StructureTensor in Algorithm 2).

    Not a Hankel tensor in the strict (symmetric) sense; see Sec. 2.3.

    Parameters
    ----------
    f
        Real array of shape ``(N1, N2)``.  Indices wrap periodically.
    patch
        Square side length ``ell`` or a pair ``(ell1, ell2)``.

    Returns
    -------
    F
        Array of shape ``(N1, N2, ell1, ell2)`` with
        ``F[m, n, i, j] = f[(m + i) mod N1, (n + j) mod N2]``.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {f.shape}")
    n1, n2 = f.shape
    ell1, ell2 = _as_patch(patch)
    if not (1 <= ell1 <= n1 and 1 <= ell2 <= n2):
        raise ValueError(
            f"need 1 <= patch <= image shape, got patch={(ell1, ell2)}, "
            f"image={f.shape}"
        )
    pad_h = ell1 - 1
    pad_w = ell2 - 1
    padded = np.pad(f, ((0, pad_h), (0, pad_w)), mode="wrap")
    s0, s1 = padded.strides
    view = as_strided(
        padded,
        shape=(n1, n2, ell1, ell2),
        strides=(s0, s1, s0, s1),
    )
    return np.ascontiguousarray(view)


def structure_add_2d(F: Array) -> Array:
    """Structure-add, StructureAdd_2: left inverse of :func:`patch_tensor`, eq. (6).

    f[m, n] = (1 / (ell1 ell2)) sum_{i,j} F[(m - i) mod N1, (n - j) mod N2, i, j]
    """
    F = np.asarray(F)
    if F.ndim != 4:
        raise ValueError(f"expected a (N1, N2, ell1, ell2) tensor, got {F.shape}")
    n1, n2, ell1, ell2 = F.shape
    out = np.zeros((n1, n2), dtype=F.dtype)
    for i in range(ell1):
        rolled_i = np.roll(F[:, :, i, :], i, axis=0)
        for j in range(ell2):
            out += np.roll(rolled_i[:, :, j], j, axis=1)
    return out / (ell1 * ell2)
