#!/usr/bin/env python3
"""Reproduce the numerical examples of Sec. 4.2 (manuscript Figs. 7-8).

Compares univariate convolutional framelets (Algorithm 1: SVD of the 1D
Hankel matrix of the vectorized image) against bivariate convolutional
framelets (Algorithm 2: truncated HOSVD of the patch tensor) at matched
compression ratio.  Outputs: ``outputs/fig1_reconstruction.png`` (Fig. 7)
and ``outputs/fig2_denoising.png`` (Fig. 8).

A sweep over integer matrix rank and HOSVD ranks chooses the 4-tuple
whose CR is closest to the matrix CR (exact when a storage count matches).
Manuscript heuristics (r = N/12 with (3N/4, 3N/4, 3ell/4, 3ell/4) for
reconstruction; r = N/50 with (N/2, N/2, ell/2, ell/2) for denoising)
are the starting matrix ranks and the tie-break preference.

Experiment 1 resizes a grayscale image to N x N and reconstructs it.
Experiment 2 denoises the same image after adding white Gaussian noise.
Defaults: N = 128, ell = N/4 = 32, noise std 80 on the 8-bit scale.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from contextlib import redirect_stderr
from dataclasses import dataclass
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from skimage import color, io, transform
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convframe.approx import (
    Approximation,
    tensor_hosvd_approx,
    vectorized_hankel_svd_approx,
)
from convframe.metrics import (
    compression_ratio,
    hosvd_storage,
    psnr,
    relative_error,
    svd_storage,
)

Array = NDArray[np.floating]

PAPER_IMAGES = (
    ROOT / "data" / "boat.png",
    ROOT / "data" / "peppers_gray.tif",
)

# Paper noise is specified on 8-bit images; we work in [0, 1].
BIT8_PEAK = 255.0


@dataclass
class ScoredApprox:
    approx: Approximation
    relative_error: float
    psnr: float
    ssim: float


@dataclass
class ImageTrial:
    name: str
    original: Array
    noisy: Array | None
    matrix: ScoredApprox
    tensor: ScoredApprox


@dataclass(frozen=True)
class RankMatch:
    """Matrix rank and HOSVD ranks with (nearly) equal compression ratio."""

    matrix_rank: int
    tensor_ranks: tuple[int, int, int, int]
    matrix_cr: float
    tensor_cr: float

    @property
    def abs_diff(self) -> float:
        return abs(self.matrix_cr - self.tensor_cr)


def parse_ranks(text: str) -> tuple[int, int, int, int]:
    parts = [int(p.strip()) for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "tensor ranks must be four integers, e.g. 96,96,24,24"
        )
    return parts[0], parts[1], parts[2], parts[3]


def as_grayscale_unit_interval(arr: np.ndarray) -> Array:
    """Convert a loaded array to a 2D float image in [0, 1]."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[-1] >= 3:
            rgb = arr[..., :3].astype(float)
            if rgb.max() > 1.5:
                rgb = rgb / BIT8_PEAK
            return np.asarray(color.rgb2gray(rgb), dtype=float)
        arr = arr[..., 0]
    elif arr.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {arr.shape}")
    arr = np.asarray(arr, dtype=float)
    if arr.max() > 1.5:
        arr = arr / BIT8_PEAK
    return arr


def load_grayscale(path: Path, size: int) -> Array:
    """Open an image, convert to grayscale in [0, 1], and resize to size x size.

    PNG and TIFF both go through ``skimage.io.imread`` so boat and peppers
    share one loader.  Bicubic resize (order=3) matches the paper notebook.
    """
    sink = StringIO()
    with warnings.catch_warnings(), redirect_stderr(sink):
        warnings.simplefilter("ignore")
        raw = io.imread(path)
    gray = as_grayscale_unit_interval(raw)
    resized = transform.resize(
        gray,
        (size, size),
        order=3,
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.clip(np.asarray(resized, dtype=float), 0.0, 1.0)


def add_gaussian_noise(image: Array, sigma_8bit: float, rng: np.random.Generator) -> Array:
    """Add N(0, sigma) noise on the 8-bit scale, working in [0, 1]."""
    sigma = float(sigma_8bit) / BIT8_PEAK
    noisy = image + rng.normal(0.0, sigma, size=image.shape)
    return np.clip(noisy, 0.0, 1.0)


def score_against(original: Array, reconstruction: Array, approx: Approximation) -> ScoredApprox:
    original = np.asarray(original, dtype=float)
    reconstruction = np.asarray(reconstruction, dtype=float)
    return ScoredApprox(
        approx=approx,
        relative_error=relative_error(original, reconstruction),
        psnr=psnr(original, reconstruction, data_range=1.0),
        ssim=float(
            structural_similarity(original, reconstruction, data_range=1.0)
        ),
    )


def heuristic_reconstruction_ranks(
    size: int, patch: int
) -> tuple[int, tuple[int, int, int, int]]:
    return size // 12, (3 * size // 4, 3 * size // 4, 3 * patch // 4, 3 * patch // 4)


def heuristic_denoising_ranks(
    size: int, patch: int
) -> tuple[int, tuple[int, int, int, int]]:
    return size // 50, (size // 2, size // 2, patch // 2, patch // 2)


def matrix_cr(size: int, patch: int, rank: int) -> float:
    n_rows = size * size
    return compression_ratio(n_rows * patch, svd_storage(n_rows, patch, rank))


def tensor_cr(size: int, patch: int, ranks: Sequence[int]) -> float:
    return compression_ratio(
        size * size * patch * patch,
        hosvd_storage(ranks, (size, size), (patch, patch)),
    )


@lru_cache(maxsize=4)
def _hosvd_storage_grid(size: int, patch: int) -> np.ndarray:
    """Integer HOSVD storage for every (r0, r1, r2, r3) in range."""
    r0 = np.arange(1, size + 1, dtype=np.int64)
    r1 = np.arange(1, size + 1, dtype=np.int64)
    r2 = np.arange(1, patch + 1, dtype=np.int64)
    r3 = np.arange(1, patch + 1, dtype=np.int64)
    return (
        r0[:, None, None, None]
        * r1[None, :, None, None]
        * r2[None, None, :, None]
        * r3[None, None, None, :]
        + size * (r0[:, None, None, None] + r1[None, :, None, None])
        + patch * (r2[None, None, :, None] + r3[None, None, None, :])
    )


def match_tensor_ranks(
    size: int,
    patch: int,
    matrix_rank: int,
    prefer: Sequence[int] | None = None,
) -> RankMatch:
    """Sweep HOSVD ranks and pick the 4-tuple whose CR matches the matrix rank.

    The search is over all integer ranks ``1..N`` (spatial modes) and
    ``1..ell`` (patch modes).  Among minimizers of ``|CR_tensor - CR_matrix|``
    it prefers balanced tuples (``r0≈r1``, ``r2≈r3``), then those closest
    to ``prefer`` if given.
    """
    n_rows = size * size
    max_rank = min(n_rows, patch)
    if not 1 <= matrix_rank <= max_rank:
        raise ValueError(
            f"matrix rank={matrix_rank} is outside 1..{max_rank} "
            f"for an ({n_rows} x {patch}) Hankel matrix"
        )
    target_store = patch * svd_storage(n_rows, patch, matrix_rank)
    store = _hosvd_storage_grid(size, patch)
    delta = np.abs(store - np.int64(target_store))
    min_d = int(delta.min())
    ii, jj, kk, ll = np.nonzero(delta == min_d)

    def score(t: int) -> tuple:
        ranks = (
            int(ii[t]) + 1,
            int(jj[t]) + 1,
            int(kk[t]) + 1,
            int(ll[t]) + 1,
        )
        balance = abs(ranks[0] - ranks[1]) + abs(ranks[2] - ranks[3])
        pref = (
            sum(abs(ranks[k] - int(prefer[k])) for k in range(4))
            if prefer is not None
            else 0
        )
        order = (ranks[0] > ranks[1], ranks[2] > ranks[3])
        return (balance, pref, order, ranks)

    best = min(range(ii.size), key=score)
    ranks = (
        int(ii[best]) + 1,
        int(jj[best]) + 1,
        int(kk[best]) + 1,
        int(ll[best]) + 1,
    )
    return RankMatch(
        matrix_rank=matrix_rank,
        tensor_ranks=ranks,
        matrix_cr=matrix_cr(size, patch, matrix_rank),
        tensor_cr=tensor_cr(size, patch, ranks),
    )


def reconstruction_ranks(size: int, patch: int) -> tuple[int, tuple[int, int, int, int]]:
    matrix_rank, prefer = heuristic_reconstruction_ranks(size, patch)
    match = match_tensor_ranks(size, patch, matrix_rank, prefer=prefer)
    return matrix_rank, match.tensor_ranks


def denoising_ranks(size: int, patch: int) -> tuple[int, tuple[int, int, int, int]]:
    matrix_rank, prefer = heuristic_denoising_ranks(size, patch)
    match = match_tensor_ranks(size, patch, matrix_rank, prefer=prefer)
    return matrix_rank, match.tensor_ranks


def run_reconstruction(
    image: Array,
    patch: int,
    matrix_rank: int,
    tensor_ranks: Sequence[int],
    name: str = "",
) -> ImageTrial:
    matrix = vectorized_hankel_svd_approx(image, patch, matrix_rank, data_range=1.0)
    tensor = tensor_hosvd_approx(image, patch, tensor_ranks, data_range=1.0)
    return ImageTrial(
        name=name,
        original=image,
        noisy=None,
        matrix=score_against(image, matrix.image, matrix),
        tensor=score_against(image, tensor.image, tensor),
    )


def run_denoising(
    image: Array,
    patch: int,
    matrix_rank: int,
    tensor_ranks: Sequence[int],
    sigma_8bit: float,
    rng: np.random.Generator,
    name: str = "",
) -> ImageTrial:
    noisy = add_gaussian_noise(image, sigma_8bit, rng)
    matrix = vectorized_hankel_svd_approx(noisy, patch, matrix_rank, data_range=1.0)
    tensor = tensor_hosvd_approx(noisy, patch, tensor_ranks, data_range=1.0)
    return ImageTrial(
        name=name,
        original=image,
        noisy=noisy,
        matrix=score_against(image, matrix.image, matrix),
        tensor=score_against(image, tensor.image, tensor),
    )


def _show_gray(ax, image: Array, title: str, lines: Sequence[str]) -> None:
    ax.imshow(np.clip(image, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    for i, line in enumerate(lines):
        ax.text(
            0.0,
            -0.02 - 0.055 * i,
            line,
            color="black",
            fontsize=8,
            ha="left",
            va="top",
            transform=ax.transAxes,
        )


def plot_reconstruction(
    trials: Sequence[ImageTrial],
    patch: int,
    matrix_rank: int,
    tensor_ranks: Sequence[int],
    out: Path,
) -> None:
    n = len(trials)
    fig, axes = plt.subplots(n, 3, figsize=(11.5, 4.6 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for row, trial in enumerate(trials):
        n1, n2 = trial.original.shape
        _show_gray(
            axes[row, 0],
            trial.original,
            "original resized image:",
            [f"shape = {n1}", f"patch_size = {patch}"],
        )
        _show_gray(
            axes[row, 1],
            trial.matrix.approx.image,
            "reconstructed image from patch matrix:",
            [
                f"rank = {matrix_rank}",
                f"compression ratio = {trial.matrix.approx.compression_ratio:.2f}",
                f"relative error = {trial.matrix.relative_error:.2f}",
            ],
        )
        _show_gray(
            axes[row, 2],
            trial.tensor.approx.image,
            "reconstructed image from patch tensor:",
            [
                f"multilinear rank = {list(tensor_ranks)}",
                f"compression ratio = {trial.tensor.approx.compression_ratio:.2f}",
                f"relative error = {trial.tensor.relative_error:.2f}",
            ],
        )
    fig.tight_layout(pad=2.4)
    fig.subplots_adjust(hspace=0.45)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_denoising(
    trials: Sequence[ImageTrial],
    patch: int,
    matrix_rank: int,
    tensor_ranks: Sequence[int],
    out: Path,
) -> None:
    n = len(trials)
    fig, axes = plt.subplots(n, 4, figsize=(15.0, 4.8 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for row, trial in enumerate(trials):
        n1, n2 = trial.original.shape
        assert trial.noisy is not None
        _show_gray(
            axes[row, 0],
            trial.original,
            "original resized image:",
            [f"shape = {n1}", f"patch_size = {patch}"],
        )
        _show_gray(
            axes[row, 1],
            trial.noisy,
            "Noisy image:",
            [f"shape = {n1}", f"patch_size = {patch}"],
        )
        _show_gray(
            axes[row, 2],
            trial.matrix.approx.image,
            "reconstructed image from patch matrix:",
            [
                f"rank = {matrix_rank}",
                f"compression ratio = {trial.matrix.approx.compression_ratio:.2f}",
                f"relative error = {trial.matrix.relative_error:.2f}",
                f"PSNR = {trial.matrix.psnr:.2f}",
                f"SSIM value = {trial.matrix.ssim:.2f}",
            ],
        )
        _show_gray(
            axes[row, 3],
            trial.tensor.approx.image,
            "reconstructed image from patch tensor:",
            [
                f"multilinear rank = {list(tensor_ranks)}",
                f"compression ratio = {trial.tensor.approx.compression_ratio:.2f}",
                f"relative error = {trial.tensor.relative_error:.2f}",
                f"PSNR = {trial.tensor.psnr:.2f}",
                f"SSIM value = {trial.tensor.ssim:.2f}",
            ],
        )
    fig.tight_layout(pad=2.4)
    fig.subplots_adjust(hspace=0.55)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _report_reconstruction(trial: ImageTrial) -> None:
    print(f"\n{trial.name}")
    print(
        f"  patch matrix  CR={trial.matrix.approx.compression_ratio:.2f}  "
        f"RE={trial.matrix.relative_error:.2f}"
    )
    print(
        f"  patch tensor  CR={trial.tensor.approx.compression_ratio:.2f}  "
        f"RE={trial.tensor.relative_error:.2f}"
    )


def _report_denoising(trial: ImageTrial) -> None:
    print(f"\n{trial.name}")
    print(
        f"  patch matrix  CR={trial.matrix.approx.compression_ratio:.2f}  "
        f"RE={trial.matrix.relative_error:.2f}  "
        f"PSNR={trial.matrix.psnr:.2f}  SSIM={trial.matrix.ssim:.2f}"
    )
    print(
        f"  patch tensor  CR={trial.tensor.approx.compression_ratio:.2f}  "
        f"RE={trial.tensor.relative_error:.2f}  "
        f"PSNR={trial.tensor.psnr:.2f}  SSIM={trial.tensor.ssim:.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        default=list(PAPER_IMAGES),
        help="grayscale images (default: data/boat.png data/peppers_gray.tif)",
    )
    parser.add_argument("--size", type=int, default=128, help="square resize (default 128)")
    parser.add_argument("--patch", type=int, default=32, help="square patch side (default 32)")
    parser.add_argument(
        "--experiment",
        choices=("reconstruction", "denoising", "both"),
        default="both",
    )
    parser.add_argument(
        "--recon-matrix-rank",
        type=int,
        default=None,
        help="matrix SVD rank (default N/12)",
    )
    parser.add_argument(
        "--recon-tensor-ranks",
        type=parse_ranks,
        default=None,
        help="HOSVD ranks r0,r1,r2,r3 (default: CR-matched to the matrix rank)",
    )
    parser.add_argument(
        "--denoise-matrix-rank",
        type=int,
        default=None,
        help="matrix SVD rank (default N/50)",
    )
    parser.add_argument(
        "--denoise-tensor-ranks",
        type=parse_ranks,
        default=None,
        help="HOSVD ranks r0,r1,r2,r3 (default: CR-matched to the matrix rank)",
    )
    parser.add_argument(
        "--match-cr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="sweep HOSVD ranks so tensor CR matches matrix CR (default on)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=80.0,
        help="Gaussian noise std on the 8-bit scale (default 80)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs",
    )
    args = parser.parse_args(argv)

    images: list[tuple[str, Array]] = []
    for path in args.images:
        path = path if path.is_absolute() else ROOT / path
        if not path.exists():
            raise FileNotFoundError(path)
        loaded = load_grayscale(path, args.size)
        images.append((path.stem, loaded))
        print(f"loaded {path}  shape={loaded.shape}  range=[{loaded.min():.2f}, {loaded.max():.2f}]")

    print(f"patch        : ({args.patch}, {args.patch})")

    recon_rank = (
        args.recon_matrix_rank
        if args.recon_matrix_rank is not None
        else args.size // 12
    )
    denoise_rank = (
        args.denoise_matrix_rank
        if args.denoise_matrix_rank is not None
        else max(1, args.size // 50)
    )
    _, recon_prefer = heuristic_reconstruction_ranks(args.size, args.patch)
    _, denoise_prefer = heuristic_denoising_ranks(args.size, args.patch)

    def _picked(
        rank: int,
        explicit: tuple[int, int, int, int] | None,
        prefer: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        if explicit is not None:
            return explicit
        if args.match_cr:
            return match_tensor_ranks(args.size, args.patch, rank, prefer=prefer).tensor_ranks
        return prefer

    recon_tensor = _picked(recon_rank, args.recon_tensor_ranks, recon_prefer)
    denoise_tensor = _picked(denoise_rank, args.denoise_tensor_ranks, denoise_prefer)

    if args.experiment in ("reconstruction", "both"):
        print("\n=== Experiment 1: reconstruction ===")
        print(f"matrix rank {recon_rank}, tensor ranks {tuple(recon_tensor)}")
        trials = []
        for name, image in images:
            print(f"reconstructing {name} ...")
            trials.append(
                run_reconstruction(
                    image,
                    args.patch,
                    recon_rank,
                    recon_tensor,
                    name=name,
                )
            )
            _report_reconstruction(trials[-1])
        out = args.out_dir / "fig1_reconstruction.png"
        plot_reconstruction(trials, args.patch, recon_rank, recon_tensor, out)
        print(f"wrote {out}")

    if args.experiment in ("denoising", "both"):
        print("\n=== Experiment 2: denoising ===")
        print(
            f"matrix rank {denoise_rank}, tensor ranks {tuple(denoise_tensor)}, "
            f"sigma={args.sigma} (8-bit), seed={args.seed}"
        )
        rng = np.random.default_rng(args.seed)
        trials = []
        for name, image in images:
            print(f"denoising {name} ...")
            trials.append(
                run_denoising(
                    image,
                    args.patch,
                    denoise_rank,
                    denoise_tensor,
                    args.sigma,
                    rng,
                    name=name,
                )
            )
            _report_denoising(trials[-1])
        out = args.out_dir / "fig2_denoising.png"
        plot_denoising(trials, args.patch, denoise_rank, denoise_tensor, out)
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
