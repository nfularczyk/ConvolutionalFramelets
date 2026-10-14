"""Circular convolution used by the framelet analysis/synthesis maps.

Definition 3.2 (2D): for periodic matrices A, B of shape (N, N),

    (A * B)[n, m] = sum_{i,j} A[i, j] B[n - i, m - j]

A local basis vector of length ell is zero-padded to length N before the
sum, equivalently implemented here by an FFT of matching size.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


def circular_conv1d(a: Array, b: Array, n: int | None = None) -> Array:
    """N-point circular convolution ``a *_N b`` of 1D arrays (Definition 2.1)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("circular_conv1d expects 1D arrays")
    length = n if n is not None else max(a.size, b.size)
    return np.fft.irfft(np.fft.rfft(a, length) * np.fft.rfft(b, length), length)


def circular_conv2d(a: Array, b: Array, shape: tuple[int, int] | None = None) -> Array:
    """Circular convolution of 2D arrays (Definition 3.2).

    Either argument may be smaller than ``shape``; it is zero-padded in the
    trailing rows/columns.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("circular_conv2d expects 2D arrays")
    if shape is None:
        n1 = max(a.shape[0], b.shape[0])
        n2 = max(a.shape[1], b.shape[1])
    else:
        n1, n2 = int(shape[0]), int(shape[1])
    return np.fft.irfft2(np.fft.rfft2(a, (n1, n2)) * np.fft.rfft2(b, (n1, n2)), (n1, n2))


def outer2(u: Array, v: Array) -> Array:
    """Rank-1 outer product u ∘ v, i.e. ``u[:, None] * v[None, :]``."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.ndim != 1 or v.ndim != 1:
        raise ValueError("outer2 expects 1D vectors")
    return np.outer(u, v)
