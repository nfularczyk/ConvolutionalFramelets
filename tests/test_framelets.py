"""Exactness and identity tests for the 1D/2D convolutional framelet maps."""

from __future__ import annotations

import numpy as np
import pytest

from convframe.approx import (
    tensor_hosvd_approx,
    vectorized_hankel_svd_approx,
)
from convframe.conv import circular_conv2d, outer2
from convframe.framelets import (
    Framelet2D,
    analyze_1d,
    analyze_2d,
    coefficient_from_convolution,
    dct_orthonormal,
    framelet_2d,
    identity_basis,
    random_orthonormal,
    reconstruct_1d,
    reconstruct_2d,
)
from convframe.hankel import (
    hankel_matrix,
    patch_tensor,
    structure_add_1d,
    structure_add_2d,
)
from convframe.hosvd import hosvd, truncated_hosvd
from convframe.metrics import compression_ratio, hosvd_storage, svd_storage


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_hankel_structure_add_1d_is_identity(rng: np.random.Generator) -> None:
    n, ell = 16, 5
    f = rng.standard_normal(n)
    rec = structure_add_1d(hankel_matrix(f, ell))
    assert rec.shape == f.shape
    np.testing.assert_allclose(rec, f, atol=1e-12)


def test_hankel_isometry_1d(rng: np.random.Generator) -> None:
    n, ell = 12, 4
    f = rng.standard_normal(n)
    F = hankel_matrix(f, ell)
    np.testing.assert_allclose(np.linalg.norm(F) ** 2, ell * np.linalg.norm(f) ** 2)


def test_hankel_structure_add_2d_is_identity(rng: np.random.Generator) -> None:
    n1, n2, ell1, ell2 = 10, 12, 3, 4
    f = rng.standard_normal((n1, n2))
    rec = structure_add_2d(patch_tensor(f, (ell1, ell2)))
    np.testing.assert_allclose(rec, f, atol=1e-12)


def test_hankel_isometry_2d(rng: np.random.Generator) -> None:
    n, ell = 9, 3
    f = rng.standard_normal((n, n))
    F = patch_tensor(f, ell)
    np.testing.assert_allclose(
        np.linalg.norm(F) ** 2, (ell * ell) * np.linalg.norm(f) ** 2
    )


def test_hankel_entries_match_definition(rng: np.random.Generator) -> None:
    n, ell = 8, 3
    f = rng.standard_normal((n, n))
    F = patch_tensor(f, ell)
    for m in range(n):
        for nn in range(n):
            for i in range(ell):
                for j in range(ell):
                    assert F[m, nn, i, j] == pytest.approx(f[(m + i) % n, (nn + j) % n])


@pytest.mark.parametrize("basis", ["identity", "dct", "random"])
def test_reconstruct_1d_exact(basis: str, rng: np.random.Generator) -> None:
    n, ell = 15, 6
    f = rng.standard_normal(n)
    if basis == "identity":
        phi, v = identity_basis(n), identity_basis(ell)
    elif basis == "dct":
        phi, v = dct_orthonormal(n), dct_orthonormal(ell)
    else:
        phi, v = random_orthonormal(n, rng), random_orthonormal(ell, rng)
    rec = reconstruct_1d(f, phi, v)
    np.testing.assert_allclose(rec, f, atol=1e-11)


@pytest.mark.parametrize("basis", ["identity", "dct", "random"])
def test_reconstruct_2d_exact(basis: str, rng: np.random.Generator) -> None:
    n, ell = 8, 3
    f = rng.standard_normal((n, n))
    if basis == "identity":
        make = identity_basis
    elif basis == "dct":
        make = dct_orthonormal
    else:
        make = lambda k: random_orthonormal(k, rng)
    rec = reconstruct_2d(f, make(n), make(n), make(ell), make(ell))
    np.testing.assert_allclose(rec, f, atol=1e-11)


def test_prop_33_coefficient_identity(rng: np.random.Generator) -> None:
    """C[j0,j1,l0,l1] equals the inner product against the convolved atom."""
    n, ell = 6, 3
    f = rng.standard_normal((n, n))
    phi_r = random_orthonormal(n, rng)
    phi_c = random_orthonormal(n, rng)
    v_r = random_orthonormal(ell, rng)
    v_c = random_orthonormal(ell, rng)
    C = analyze_2d(f, phi_r, phi_c, v_r, v_c)
    # Check a handful of index tuples, not the full N^2 ell^2 grid.
    for j0, j1, l0, l1 in [(0, 0, 0, 0), (1, 2, 0, 1), (5, 3, 2, 2)]:
        inner = coefficient_from_convolution(
            f, phi_r[:, j0], phi_c[:, j1], v_r[:, l0], v_c[:, l1]
        )
        assert C[j0, j1, l0, l1] == pytest.approx(inner, abs=1e-10)


def test_framelet_atom_is_convolution(rng: np.random.Generator) -> None:
    n, ell = 7, 3
    phi = rng.standard_normal(n)
    psi = rng.standard_normal(n)
    v = rng.standard_normal(ell)
    w = rng.standard_normal(ell)
    atom = framelet_2d(phi, psi, v, w, (n, n))
    expected = circular_conv2d(outer2(phi, psi), outer2(v, w), (n, n))
    np.testing.assert_allclose(atom, expected, atol=1e-12)


def test_full_hosvd_reconstructs_tensor(rng: np.random.Generator) -> None:
    X = rng.standard_normal((5, 6, 3, 4))
    rec = truncated_hosvd(X, X.shape)
    np.testing.assert_allclose(rec, X, atol=1e-11)


def test_framelet2d_from_full_hosvd_is_exact(rng: np.random.Generator) -> None:
    n, ell = 8, 3
    f = rng.standard_normal((n, n))
    model, core = Framelet2D.from_hosvd(f, ell)
    rec = model.reconstruct_from(core)
    np.testing.assert_allclose(rec, f, atol=1e-11)


def test_truncated_hosvd_error_decreases(rng: np.random.Generator) -> None:
    n, ell = 8, 4
    f = rng.standard_normal((n, n))
    coarse = tensor_hosvd_approx(f, ell, (2, 2, 1, 1))
    fine = tensor_hosvd_approx(f, ell, (6, 6, 3, 3))
    assert fine.relative_error < coarse.relative_error


def test_vectorized_hankel_full_rank_is_exact(rng: np.random.Generator) -> None:
    n, ell = 6, 3
    f = rng.standard_normal((n, n))
    rec = vectorized_hankel_svd_approx(f, ell, rank=ell)
    np.testing.assert_allclose(rec.image, f, atol=1e-11)
    assert rec.relative_error < 1e-12


def test_paper_reconstruction_compression_ratios() -> None:
    """Storage counts for manuscript Fig. 7 (reconstruction)."""
    n, ell, rank = 128, 32, 10
    n_rows, n_cols = n * n, ell
    matrix_cr = compression_ratio(n_rows * n_cols, svd_storage(n_rows, n_cols, rank))
    tensor_cr = compression_ratio(
        n * n * ell * ell,
        hosvd_storage((96, 96, 24, 24), (n, n), (ell, ell)),
    )
    assert matrix_cr == pytest.approx(3.193567643296583)
    assert tensor_cr == pytest.approx(3.1450235147326997)


def test_paper_denoising_compression_ratios() -> None:
    """Storage counts for manuscript Fig. 8 (denoising)."""
    n, ell, rank = 128, 32, 2
    n_rows, n_cols = n * n, ell
    matrix_cr = compression_ratio(n_rows * n_cols, svd_storage(n_rows, n_cols, rank))
    tensor_cr = compression_ratio(
        n * n * ell * ell,
        hosvd_storage((64, 64, 16, 16), (n, n), (ell, ell)),
    )
    assert matrix_cr == pytest.approx(15.967838216482914)
    assert tensor_cr == pytest.approx(15.738712776176753)


def test_analyze_1d_matches_phiT_F_v(rng: np.random.Generator) -> None:
    n, ell = 10, 4
    f = rng.standard_normal(n)
    phi = dct_orthonormal(n)
    v = dct_orthonormal(ell)
    C = analyze_1d(f, phi, v)
    np.testing.assert_allclose(C, phi.T @ hankel_matrix(f, ell) @ v, atol=1e-12)


def test_hosvd_core_is_mode_products(rng: np.random.Generator) -> None:
    X = rng.standard_normal((4, 5, 3, 3))
    core, factors = hosvd(X)
    # Reconstructing via the factors must recover X (full rank).
    rec = core
    from convframe.hosvd import mode_n_product

    for mode, U in enumerate(factors):
        rec = mode_n_product(rec, U, mode)
    np.testing.assert_allclose(rec, X, atol=1e-11)
