"""Mode-n products and the higher-order SVD (Tucker / HOSVD).

Definitions follow Appendix A (0-based modes, ``×_n``, ``X_(n)``).  By
Proposition 3.1 the HOSVD of the patch tensor supplies data-adapted
orthonormal bases for 2D convolutional framelets (Algorithm 2).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


def unfold(X: Array, mode: int) -> Array:
    """Mode-n matricization X_(n): mode axis becomes the row axis.

    Column ordering is NumPy C order, which differs from the ordering in
    Appendix A.2; this does not affect the left singular vectors.
    """
    X = np.asarray(X)
    if not 0 <= mode < X.ndim:
        raise ValueError(f"mode {mode} out of range for ndim={X.ndim}")
    return np.moveaxis(X, mode, 0).reshape(X.shape[mode], -1)


def fold(Xn: Array, mode: int, shape: Sequence[int]) -> Array:
    """Inverse of :func:`unfold`."""
    shape = tuple(shape)
    leading = [shape[mode]]
    trailing = [s for i, s in enumerate(shape) if i != mode]
    return np.moveaxis(np.asarray(Xn).reshape(leading + trailing), 0, mode)


def mode_n_product(X: Array, U: Array, mode: int) -> Array:
    """n-mode product ``X ×_n U`` (Appendix A.1).

    ``U`` has shape ``(J, I_n)`` and contracts the ``mode``-th axis of ``X``::

        (X ×_n U)[..., j, ...] = sum_{i} X[..., i, ...] U[j, i]
    """
    X = np.asarray(X)
    U = np.asarray(U)
    if U.ndim != 2:
        raise ValueError(f"U must be a matrix, got shape {U.shape}")
    if U.shape[1] != X.shape[mode]:
        raise ValueError(
            f"U has {U.shape[1]} columns but mode-{mode} of X has "
            f"length {X.shape[mode]}"
        )
    new_shape = list(X.shape)
    new_shape[mode] = U.shape[0]
    return fold(U @ unfold(X, mode), mode, new_shape)


def multi_mode_product(X: Array, factors: Sequence[Array]) -> Array:
    """Apply ``X ×_0 U^{(0)} ×_1 U^{(1)} ⋯ ×_{d-1} U^{(d-1)}``."""
    if len(factors) != X.ndim:
        raise ValueError(
            f"need one factor per mode, got {len(factors)} factors for "
            f"ndim={X.ndim}"
        )
    Y = np.asarray(X, dtype=float)
    for mode, U in enumerate(factors):
        Y = mode_n_product(Y, U, mode)
    return Y


def hosvd(X: Array, ranks: Sequence[int] | None = None) -> tuple[Array, list[Array]]:
    """Higher-order SVD of ``X``.

    For each mode n the left singular vectors of the mode-n unfolding are
    collected as a unitary factor ``U^{(n)}``.  The core is

        S = X ×_0 U^{(0)T} ×_1 ⋯ ×_{d-1} U^{(d-1)T}

    (Proposition 3.1, Algorithm 2).  If ``ranks`` is given, each factor is truncated to the
    leading ``ranks[n]`` singular vectors (truncated HOSVD).
    """
    X = np.asarray(X, dtype=float)
    if ranks is None:
        ranks = X.shape
    if len(ranks) != X.ndim:
        raise ValueError(f"ranks must have length {X.ndim}, got {ranks}")
    factors: list[Array] = []
    for mode, rank in enumerate(ranks):
        if not 1 <= rank <= X.shape[mode]:
            raise ValueError(
                f"rank[{mode}]={rank} is outside 1..{X.shape[mode]}"
            )
        Un, _, _ = np.linalg.svd(unfold(X, mode), full_matrices=False)
        factors.append(np.ascontiguousarray(Un[:, :rank]))
    core = multi_mode_product(X, [U.T for U in factors])
    return core, factors


def truncated_hosvd(X: Array, ranks: Sequence[int]) -> Array:
    """Reconstruct the truncated HOSVD approximation of ``X``."""
    core, factors = hosvd(X, ranks)
    return multi_mode_product(core, factors)
