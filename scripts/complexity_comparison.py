#!/usr/bin/env python3
"""Flop counts and wall-clock timings: 1D vs 2D convolutional framelets.

Produces Tables 1-2 (Secs. 4.2.1-4.2.2) and the flop model of Sec. 4.1.
Stages match ``vectorized_hankel_svd_approx`` (Algorithm 1) and
``tensor_hosvd_approx`` (Algorithm 2) in ``convframe.approx``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from convframe.framelets import Framelet2D
from convframe.hankel import hankel_matrix, patch_tensor, structure_add_1d, structure_add_2d
from convframe.hosvd import hosvd, multi_mode_product
from convframe.metrics import hosvd_storage, svd_storage

import ConvFrameEX as ex

def svd_flops(m: int, n: int) -> float:
    """R-SVD flop count C_SVD(m, n), eq. (8)."""
    k = min(m, n)
    big = max(m, n)
    return 4.0 * big * k * k - 4.0 * k * k * k / 3.0


def gemm_flops(m: int, n: int, k: int) -> float:
    """C = A_{m x k} @ B_{k x n} costs 2mnk flops (Lemma 4.1)."""
    return 2.0 * m * n * k


@dataclass
class FlopBreakdown:
    label: str
    lift: float
    factorize: float
    form_approx: float
    invert: float

    @property
    def total(self) -> float:
        return self.lift + self.factorize + self.form_approx + self.invert


def matrix_flops(size: int, patch: int, rank: int) -> FlopBreakdown:
    """Flop model T_1D of Theorem 4.4, eq. (9).

    H is (N^2 x ell).  ``np.linalg.svd(..., full_matrices=False)`` computes
    the full economy SVD, then the code truncates to rank r.
    """
    n_rows = size * size
    # Lifting is a strided copy: 0 flops.
    lift = 0.0
    factorize = svd_flops(n_rows, patch)
    # (U[:, :r] * s[:r]) is n_rows * r, then @ Vt[:r, :] is n_rows x r  times r x ell
    form_approx = n_rows * rank + gemm_flops(n_rows, patch, rank)
    # structure-add (Proposition 4.3): ell adds of a length-N^2 vector, plus a scale
    invert = 2.0 * n_rows * patch
    return FlopBreakdown("1D framelets", lift, factorize, form_approx, invert)


def tensor_flops(
    size: int, patch: int, ranks: Sequence[int]
) -> FlopBreakdown:
    """Flop model T_2D of Theorem 4.10, eq. (16).

    Classical (not sequentially truncated) HOSVD: each mode-n SVD is taken
    on an unfolding of the full patch tensor, eq. (11).
    """
    r0, r1, r2, r3 = (int(r) for r in ranks)
    n, ell = size, patch
    lift = 0.0
    # unfoldings: (N, N ell^2), (N, N ell^2), (ell, N^2 ell), (ell, N^2 ell)
    factorize = (
        2.0 * svd_flops(n, n * ell * ell) + 2.0 * svd_flops(ell, n * n * ell)
    )
    # core: S = H x_0 U0^T x_1 U1^T x_2 U2^T x_3 U3^T, sequential as in multi_mode_product
    core = (
        gemm_flops(r0, n * ell * ell, n)
        + gemm_flops(r1, r0 * ell * ell, n)
        + gemm_flops(r2, r0 * r1 * ell, ell)
        + gemm_flops(r3, r0 * r1 * r2, ell)
    )
    # synthesis, eq. (14): S x_0 U0 x_1 U1 x_2 U2 x_3 U3
    synth = (
        gemm_flops(n, r1 * r2 * r3, r0)
        + gemm_flops(n, n * r2 * r3, r1)
        + gemm_flops(ell, n * n * r3, r2)
        + gemm_flops(ell, n * n * ell, r3)
    )
    form_approx = core + synth
    invert = 2.0 * n * n * ell * ell  # structure-add, Proposition 4.6
    return FlopBreakdown("2D framelets", lift, factorize, form_approx, invert)


def _time_call(fn: Callable[[], object], warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    arr = np.asarray(samples)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def time_1d(image: np.ndarray, patch: int, rank: int, warmup: int, repeats: int) -> dict:
    vec = np.ascontiguousarray(image.reshape(-1))

    def lift():
        return hankel_matrix(vec, patch)

    H = lift()

    def factorize():
        return np.linalg.svd(H, full_matrices=False)

    U, s, Vt = factorize()

    def form():
        return (U[:, :rank] * s[:rank]) @ Vt[:rank, :]

    H_hat = form()

    def invert():
        return structure_add_1d(H_hat).reshape(image.shape)

    stages = {
        "lift": _time_call(lift, warmup, repeats),
        "factorize": _time_call(factorize, warmup, repeats),
        "form_approx": _time_call(form, warmup, repeats),
        "invert": _time_call(invert, warmup, repeats),
    }

    def total():
        Hh = hankel_matrix(vec, patch)
        U_, s_, Vt_ = np.linalg.svd(Hh, full_matrices=False)
        rec = structure_add_1d((U_[:, :rank] * s_[:rank]) @ Vt_[:rank, :])
        return rec

    stages["total"] = _time_call(total, warmup, repeats)
    return {k: {"mean_s": v[0], "std_s": v[1]} for k, v in stages.items()}


def time_2d(
    image: np.ndarray, patch: int, ranks: Sequence[int], warmup: int, repeats: int
) -> dict:
    def lift():
        return patch_tensor(image, patch)

    F = lift()

    def factorize():
        return hosvd(F, ranks)

    core, factors = factorize()

    def form():
        return multi_mode_product(core, factors)

    F_hat = form()

    def invert():
        return structure_add_2d(F_hat)

    stages = {
        "lift": _time_call(lift, warmup, repeats),
        "factorize": _time_call(factorize, warmup, repeats),
        "form_approx": _time_call(form, warmup, repeats),
        "invert": _time_call(invert, warmup, repeats),
    }

    def total():
        model, c = Framelet2D.from_hosvd(image, patch, ranks)
        return model.reconstruct_from(c)

    stages["total"] = _time_call(total, warmup, repeats)
    return {k: {"mean_s": v[0], "std_s": v[1]} for k, v in stages.items()}


def fmt_flops(x: float) -> str:
    if x >= 1e9:
        return f" {x / 1e9:8.2f} GFLOP"
    if x >= 1e6:
        return f" {x / 1e6:8.2f} MFLOP"
    return f" {x / 1e3:8.2f} kFLOP"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--patch", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args(argv)

    recon_r, recon_t = ex.reconstruction_ranks(args.size, args.patch)
    denoise_r, denoise_t = ex.denoising_ranks(args.size, args.patch)

    image = np.random.default_rng(args.seed).random((args.size, args.size))

    print(f"host     : {platform.platform()}")
    print(f"numpy    : {np.__version__}")
    print(f"N x N    : {args.size} x {args.size},  ell = {args.patch}")
    print(f"1D rank  : recon {recon_r}, denoise {denoise_r}")
    print(f"2D ranks : recon {list(recon_t)}, denoise {list(denoise_t)}")

    results: dict = {
        "host": platform.platform(),
        "numpy": np.__version__,
        "size": args.size,
        "patch": args.patch,
        "recon": {"matrix_rank": recon_r, "tensor_ranks": list(recon_t)},
        "denoise": {"matrix_rank": denoise_r, "tensor_ranks": list(denoise_t)},
    }

    for name, rank, tens in (
        ("reconstruction", recon_r, recon_t),
        ("denoising", denoise_r, denoise_t),
    ):
        f1 = matrix_flops(args.size, args.patch, rank)
        f2 = tensor_flops(args.size, args.patch, tens)
        print(f"\n=== {name} flop model ===")
        print(f"{'stage':<16} {'1D framelets':>16} {'2D framelets':>16} {'ratio':>10}")
        for stage in ("factorize", "form_approx", "invert", "total"):
            a = getattr(f1, stage) if stage != "total" else f1.total
            b = getattr(f2, stage) if stage != "total" else f2.total
            ratio = b / a if a else float("inf")
            print(f"{stage:<16} {fmt_flops(a):>16} {fmt_flops(b):>16} {ratio:9.1f}x")
        results[name] = {
            "flops_1d": asdict(f1) | {"total": f1.total},
            "flops_2d": asdict(f2) | {"total": f2.total},
            "storage_1d": svd_storage(args.size * args.size, args.patch, rank),
            "storage_2d": hosvd_storage(tens, (args.size, args.size), (args.patch, args.patch)),
        }

    print("\n=== wall-clock (reconstruction ranks) ===")
    t1 = time_1d(image, args.patch, recon_r, args.warmup, args.repeats)
    t2 = time_2d(image, args.patch, recon_t, args.warmup, args.repeats)
    print(f"{'stage':<16} {'1D mean (s)':>14} {'2D mean (s)':>14} {'ratio':>10}")
    for stage in ("lift", "factorize", "form_approx", "invert", "total"):
        a, b = t1[stage]["mean_s"], t2[stage]["mean_s"]
        ratio = b / a if a > 0 else float("inf")
        print(f"{stage:<16} {a:14.4f} {b:14.4f} {ratio:9.1f}x")
    results["timings_reconstruction"] = {"1d": t1, "2d": t2}

    print("\n=== wall-clock (denoising ranks) ===")
    t1d = time_1d(image, args.patch, denoise_r, args.warmup, args.repeats)
    t2d = time_2d(image, args.patch, denoise_t, args.warmup, args.repeats)
    print(f"{'stage':<16} {'1D mean (s)':>14} {'2D mean (s)':>14} {'ratio':>10}")
    for stage in ("lift", "factorize", "form_approx", "invert", "total"):
        a, b = t1d[stage]["mean_s"], t2d[stage]["mean_s"]
        ratio = b / a if a > 0 else float("inf")
        print(f"{stage:<16} {a:14.4f} {b:14.4f} {ratio:9.1f}x")
    results["timings_denoising"] = {"1d": t1d, "2d": t2d}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "complexity_comparison.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
