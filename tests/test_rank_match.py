"""CR-matching rank sweep used by the numerical-results experiments."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
scripts = ROOT / "scripts"
if str(scripts) not in sys.path:
    sys.path.insert(0, str(scripts))

import ConvFrameEX as ex  # noqa: E402


def test_match_small_grid_crs_agree() -> None:
    match = ex.match_tensor_ranks(16, 4, 2)
    assert match.matrix_rank == 2
    assert round(match.matrix_cr, 2) == round(match.tensor_cr, 2)
    assert match.abs_diff == pytest.approx(0.0, abs=5e-3)


def test_paper_reconstruction_ranks_match_two_decimals() -> None:
    rank, tens = ex.reconstruction_ranks(128, 32)
    assert rank == 10
    assert round(ex.matrix_cr(128, 32, rank), 2) == round(
        ex.tensor_cr(128, 32, tens), 2
    )


def test_paper_denoising_ranks_match_two_decimals() -> None:
    rank, tens = ex.denoising_ranks(128, 32)
    assert rank == 2
    assert round(ex.matrix_cr(128, 32, rank), 2) == round(
        ex.tensor_cr(128, 32, tens), 2
    )



def test_heuristic_starters_differ_from_matched() -> None:
    """Heuristic 4-tuples are starters; matched ranks are the paper values."""
    _, recon_h = ex.heuristic_reconstruction_ranks(128, 32)
    _, denoise_h = ex.heuristic_denoising_ranks(128, 32)
    assert recon_h == (96, 96, 24, 24)
    assert denoise_h == (64, 64, 16, 16)
    _, recon_m = ex.reconstruction_ranks(128, 32)
    _, denoise_m = ex.denoising_ranks(128, 32)
    assert recon_m != recon_h
    assert denoise_m != denoise_h
