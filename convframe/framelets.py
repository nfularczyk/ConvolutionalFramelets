"""Convolutional framelets: analysis, synthesis, and reconstruction.

1D  (Proposition 2.4, eq. (4))
    f = (1 / ell) sum_{i,j} <f, v_j * phi_i>  (v_j * phi_i)

2D  (Theorem 3.4)
    f = (1 / ell^2) sum_{j_0, j_1, l_0, l_1}
            <f, (phi_{j_0} ∘ phi_{j_1}) * (v_{l_0} ∘ v_{l_1})>
            (phi_{j_0} ∘ phi_{j_1}) * (v_{l_0} ∘ v_{l_1})

The sums are never formed entrywise.  By Proposition 3.3 the coefficient
tensor is the coordinate expansion of the patch tensor, so analysis is
four mode-n products and synthesis is the inverse mode-n products followed
by structure-add (eq. (6)).  The convolution form is kept for verification
and for evaluating individual framelet atoms.

Code names: ``phi_row, phi_col`` are the global bases {phi_{j_0}}, {phi_{j_1}};
``v_row, v_col`` are the local bases {v_{l_0}}, {v_{l_1}}.  Index arguments
``j0, j1, l0, l1`` are the manuscript's ``j_0, j_1, l_0, l_1`` (0-based).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct

from .conv import circular_conv1d, circular_conv2d, outer2
from .hankel import hankel_matrix, patch_tensor, structure_add_1d, structure_add_2d
from .hosvd import hosvd, multi_mode_product

Array = NDArray[np.floating]
BasisPair = Tuple[Array, Array]


def identity_basis(n: int) -> Array:
    """Canonical Euclidean basis (the local/global-basis special cases of Sec. 2.2)."""
    return np.eye(n, dtype=float)


def dct_orthonormal(n: int) -> Array:
    """Orthonormal DCT-II matrix whose columns are the basis vectors."""
    return dct(np.eye(n), type=2, norm="ortho", axis=0)


def random_orthonormal(
    n: int, rng: np.random.Generator | None = None
) -> Array:
    """Haar-random orthonormal matrix (QR of a Gaussian draw)."""
    if rng is None:
        rng = np.random.default_rng()
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    q *= np.sign(np.diag(r))
    return q


def _check_onb(U: Array, n: int, name: str) -> Array:
    U = np.asarray(U, dtype=float)
    if U.shape != (n, n):
        raise ValueError(f"{name} must have shape {(n, n)}, got {U.shape}")
    gram = U.T @ U
    if not np.allclose(gram, np.eye(n), atol=1e-8):
        raise ValueError(f"{name} columns are not orthonormal")
    return U


def analyze_1d(f: Array, phi: Array, v: Array) -> Array:
    """Coefficient matrix C[i, j] = <f, v_j * phi_i> (Proposition 2.2, eq. (3)).

    Equivalently C = Phi^T F V, where F is the patch matrix of f.
    """
    f = np.asarray(f, dtype=float)
    n = f.shape[0]
    ell = v.shape[0]
    phi = _check_onb(phi, n, "phi")
    v = _check_onb(v, ell, "v")
    F = hankel_matrix(f, ell)
    return phi.T @ F @ v


def synthesize_1d(C: Array, phi: Array, v: Array) -> Array:
    """Rebuild the patch matrix F = Phi C V^T from its coefficients (Remark 2.3)."""
    return phi @ C @ v.T


def reconstruct_1d(f: Array, phi: Array, v: Array) -> Array:
    """Exact 1D reconstruction (Proposition 2.4)."""
    C = analyze_1d(f, phi, v)
    return structure_add_1d(synthesize_1d(C, phi, v))


def framelet_1d(phi_i: Array, v_j: Array, n: int) -> Array:
    """Single 1D atom v_j * phi_i, with v_j zero-padded to length n."""
    return circular_conv1d(phi_i, v_j, n)


def analyze_2d(
    f: Array,
    phi_row: Array,
    phi_col: Array,
    v_row: Array,
    v_col: Array,
) -> Array:
    """Coefficient tensor of the 2D convolutional framelet (Proposition 3.3).

    C[j_0, j_1, l_0, l_1] = <f, (phi_{j_0} ∘ phi_{j_1}) * (v_{l_0} ∘ v_{l_1})>
                          = <F,  phi_{j_0} ∘ phi_{j_1} ∘ v_{l_0} ∘ v_{l_1}>

    Computed as C = F ×_0 Phi_row^T ×_1 Phi_col^T ×_2 V_row^T ×_3 V_col^T.
    """
    f = np.asarray(f, dtype=float)
    if f.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {f.shape}")
    n1, n2 = f.shape
    ell1, ell2 = v_row.shape[0], v_col.shape[0]
    phi_row = _check_onb(phi_row, n1, "phi_row")
    phi_col = _check_onb(phi_col, n2, "phi_col")
    v_row = _check_onb(v_row, ell1, "v_row")
    v_col = _check_onb(v_col, ell2, "v_col")
    F = patch_tensor(f, (ell1, ell2))
    return multi_mode_product(F, [phi_row.T, phi_col.T, v_row.T, v_col.T])


def synthesize_2d(
    C: Array,
    phi_row: Array,
    phi_col: Array,
    v_row: Array,
    v_col: Array,
) -> Array:
    """Rebuild the patch tensor from its framelet coefficients."""
    return multi_mode_product(C, [phi_row, phi_col, v_row, v_col])


def reconstruct_2d(
    f: Array,
    phi_row: Array,
    phi_col: Array,
    v_row: Array,
    v_col: Array,
) -> Array:
    """Exact 2D reconstruction (Theorem 3.4)."""
    C = analyze_2d(f, phi_row, phi_col, v_row, v_col)
    return structure_add_2d(synthesize_2d(C, phi_row, phi_col, v_row, v_col))


def framelet_2d(
    phi_j0: Array,
    phi_j1: Array,
    v_l0: Array,
    v_l1: Array,
    shape: Tuple[int, int],
) -> Array:
    """Single 2D atom (phi_{j_0} ∘ phi_{j_1}) * (v_{l_0} ∘ v_{l_1}), eq. (7)."""
    return circular_conv2d(outer2(phi_j0, phi_j1), outer2(v_l0, v_l1), shape)


def coefficient_from_convolution(
    f: Array,
    phi_j0: Array,
    phi_j1: Array,
    v_l0: Array,
    v_l1: Array,
) -> float:
    """Inner-product form of a single coefficient (Proposition 3.3)."""
    atom = framelet_2d(phi_j0, phi_j1, v_l0, v_l1, f.shape)
    return float(np.vdot(f, atom))


@dataclass
class Framelet2D:
    """2D convolutional framelets with a fixed choice of bases.

    Use :meth:`identity` or :meth:`dct` for fixed bases (cf. Sec. 2.2), or
    :meth:`from_hosvd` for the data-adapted HOSVD bases (Proposition 3.1,
    Algorithm 2).

    Parameters
    ----------
    phi_row, phi_col
        Global orthonormal bases for the two spatial modes, each of
        shape ``(N, N)`` (or ``(N, r)`` after truncation).
    v_row, v_col
        Local orthonormal bases for the two patch modes, each of shape
        ``(ell, ell)`` (or ``(ell, r)`` after truncation).
    """

    phi_row: Array
    phi_col: Array
    v_row: Array
    v_col: Array

    @classmethod
    def identity(cls, n1: int, n2: int, ell1: int, ell2: int | None = None) -> "Framelet2D":
        if ell2 is None:
            ell2 = ell1
        return cls(
            identity_basis(n1),
            identity_basis(n2),
            identity_basis(ell1),
            identity_basis(ell2),
        )

    @classmethod
    def dct(cls, n1: int, n2: int, ell1: int, ell2: int | None = None) -> "Framelet2D":
        if ell2 is None:
            ell2 = ell1
        return cls(
            dct_orthonormal(n1),
            dct_orthonormal(n2),
            dct_orthonormal(ell1),
            dct_orthonormal(ell2),
        )

    @classmethod
    def from_hosvd(
        cls,
        f: Array,
        patch: int | Tuple[int, int],
        ranks: Sequence[int] | None = None,
    ) -> tuple["Framelet2D", Array]:
        """HOSVD bases of the patch tensor, plus the core tensor S.

        If ``ranks`` is omitted the factors are square (exact reconstruction).
        A 4-tuple ``(r_0, r_1, r_2, r_3)`` yields the truncated representation
        used in Sec. 4.2.
        """
        if isinstance(patch, tuple):
            ell1, ell2 = patch
        else:
            ell1 = ell2 = int(patch)
        F = patch_tensor(f, (ell1, ell2))
        core, factors = hosvd(F, ranks)
        model = cls(factors[0], factors[1], factors[2], factors[3])
        return model, core

    @property
    def patch(self) -> Tuple[int, int]:
        return int(self.v_row.shape[0]), int(self.v_col.shape[0])

    def analyze(self, f: Array) -> Array:
        """Framelet coefficients of ``f``.

        Unlike :func:`analyze_2d`, this does not require square ONBs: thin
        HOSVD factors are allowed, and the result is then a compressed core
        (approximate reconstruction).
        """
        f = np.asarray(f, dtype=float)
        ell1, ell2 = self.patch
        F = patch_tensor(f, (ell1, ell2))
        return multi_mode_product(
            F, [self.phi_row.T, self.phi_col.T, self.v_row.T, self.v_col.T]
        )

    def synthesize(self, C: Array) -> Array:
        """Patch tensor from coefficients."""
        return multi_mode_product(
            C, [self.phi_row, self.phi_col, self.v_row, self.v_col]
        )

    def reconstruct(self, f: Array) -> Array:
        """Analyze ``f`` and invert.  Exact when the factors are square ONBs."""
        return structure_add_2d(self.synthesize(self.analyze(f)))

    def reconstruct_from(self, C: Array) -> Array:
        """Invert a coefficient tensor already in hand."""
        return structure_add_2d(self.synthesize(C))

    def atom(self, j0: int, j1: int, l0: int, l1: int, shape: Tuple[int, int]) -> Array:
        """The framelet ``(phi_{j_0} ∘ phi_{j_1}) * (v_{l_0} ∘ v_{l_1})``, eq. (7)."""
        return framelet_2d(
            self.phi_row[:, j0],
            self.phi_col[:, j1],
            self.v_row[:, l0],
            self.v_col[:, l1],
            shape,
        )
