# Convolutional Framelets and Tensor Representation

Code and data for *Convolutional Framelets and Tensor Representation*

## Environment

```bash
conda env create -f environment.yml
conda activate convFrame
```

Run commands from the repository root.

## Package

`convframe/` implements the methods in *Convolutional Framelets and Tensor Representation*. 

Equation, section, and figure numbers in docstrings refer to the manuscript.


The following is an example of reconstructing a random 2D signal using a basis chosen for each mode:
```python
import numpy as np
from convframe import reconstruct_2d, dct_orthonormal

f = np.random.default_rng(0).standard_normal((32, 32))
ell = 8
rec = reconstruct_2d(
    f,
    dct_orthonormal(32), dct_orthonormal(32),
    dct_orthonormal(ell), dct_orthonormal(ell),
)
assert np.allclose(rec, f)  # full bases give exact reconstruction
```

## Numerical experiments (Sec. 4.2, Figs. 7–8)

```bash
conda activate convFrame
python scripts/ConvFrameEX.py
```

Uses `data/boat.png` and `data/peppers_gray.tif` (resized to 128×128, patch
32×32). Images load as grayscale in `[0, 1]`; noise σ is specified on the
8-bit scale and scaled accordingly.

- 1D (Algorithm 1): SVD of the 1D Hankel matrix of the vectorized image
  (`vectorized_hankel_svd_approx`)
- 2D (Algorithm 2): truncated HOSVD of the patch tensor (`tensor_hosvd_approx`)
- Tensor ranks are chosen to match the 1D compression ratio to two decimal places
  (`--no-match-cr` uses the manually found heuristics)

Writes `outputs/fig1_reconstruction.png` (paper Fig. 7),
and `outputs/fig2_denoising.png` (paper Fig. 8). 

## Jupyter Notebook

The Jupyter notebook is an interactive way to reproduce the numerical experiments above while also modifying parameters.
Notebook: `notebooks/ConvFrameEX.ipynb`

```bash
jupyter lab notebooks/ConvFrameEX.ipynb
```

Note that running the notebook overwrites `outputs/fig1_reconstruction.png` and `outputs/fig2_denoising.png`.

## Complexity (Fig. 6, Sec. 4.1; Tables 1–2, Sec. 4.2)

```bash
conda activate convFrame
python scripts/complexity_comparison.py      # Tables 1–2
python scripts/plot_complexity_scaling.py    # Fig. 6 (outputs/fig_complexity_scaling.png)
```

## Reproducing the Numerical Results

All commands run from the repository root after `conda activate convFrame`.

| Script / notebook | Output file | Paper item | Command | Seed | Expected key numbers |
|---|---|---|---|---|---|
| `scripts/ConvFrameEX.py` | `outputs/fig1_reconstruction.png` | Fig. 7 (reconstruction, Sec. 4.2) | `python scripts/ConvFrameEX.py --experiment reconstruction` | none (deterministic) | N=128, ℓ=32; ranks r=10 vs (95,104,23,23); CR 3.19, RE boat 0.0723 (1D) vs 0.0417 (2D); peppers 0.08 vs 0.03 |
| `scripts/ConvFrameEX.py` | `outputs/fig2_denoising.png` | Fig. 8 (denoising, Sec. 4.2) | `python scripts/ConvFrameEX.py --experiment denoising` | `--seed 0`, σ=80 (8-bit) | ranks r=2 vs (48,50,18,24); CR 15.97, Boat PSNR 19.72 vs 19.99, SSIM 0.37 vs 0.38; peppers PSNR 17.21 vs 19.78, SSIM 0.31 vs 0.50 |
| `scripts/plot_complexity_scaling.py` | `outputs/fig_complexity_scaling.png` | Fig. 6 | `python scripts/plot_complexity_scaling.py` | none | T_2D/T_1D = 413.2 at N=128, 834.6 at N=256 |
| `scripts/complexity_comparison.py` | stdout + `outputs/complexity_comparison.json` | Tables 1–2 (Secs. 4.2.1–4.2.2) | `python scripts/complexity_comparison.py` | `--seed 0` (random test image) | Table 1 totals: reconstruction 78.76 MFLOP (1D) vs 32.55 GFLOP (2D), 413×; denoising 70.24 M vs 26.13 G, 372×. Table 2 wall-clock depends on hardware (about 360× end-to-end on an M-series Mac) |
| `notebooks/ConvFrameEX.ipynb` | same as `ConvFrameEX.py` (Figs. 7–8) | Figs. 7–8 | `jupyter lab notebooks/ConvFrameEX.ipynb` | `SEED = 0` | same as above |

Running `python scripts/ConvFrameEX.py` without flags produces both Fig. 7 and Fig. 8.

## Tests

```bash
conda activate convFrame
pytest
```

## Layout

| Path | Role |
|------|------|
| `convframe/` | Library |
| `scripts/ConvFrameEX.py` | Numerical experiments |
| `scripts/complexity_comparison.py` | Flop counts and wall-clock timings (Tables 1–2) |
| `scripts/plot_complexity_scaling.py` | Flop-model scaling (Fig. 6) |
| `notebooks/ConvFrameEX.ipynb` | Same experiments, editable |
| `data/` | `boat.png`, `peppers_gray.tif` |
| `outputs/` | Figures |
| `tests/` | Exactness and CR checks |
