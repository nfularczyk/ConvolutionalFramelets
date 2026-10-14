"""Two-dimensional convolutional framelets.

Companion code for *Convolutional Framelets and Tensor Representation*
(Conti, Fularczyk, Labate).  Equation, section, and figure numbers in the
docstrings refer to that manuscript.
"""

from .approx import (
    Approximation,
    tensor_hosvd_approx,
    vectorized_hankel_svd_approx,
)
from .conv import circular_conv1d, circular_conv2d, outer2
from .framelets import (
    Framelet2D,
    analyze_1d,
    analyze_2d,
    dct_orthonormal,
    identity_basis,
    random_orthonormal,
    reconstruct_1d,
    reconstruct_2d,
    synthesize_1d,
    synthesize_2d,
)
from .hankel import (
    hankel_matrix,
    patch_tensor,
    structure_add_1d,
    structure_add_2d,
)
from .hosvd import hosvd, mode_n_product, truncated_hosvd, unfold
from .metrics import compression_ratio, psnr, relative_error

__all__ = [
    "Approximation",
    "Framelet2D",
    "tensor_hosvd_approx",
    "vectorized_hankel_svd_approx",
    "analyze_1d",
    "analyze_2d",
    "circular_conv1d",
    "circular_conv2d",
    "compression_ratio",
    "dct_orthonormal",
    "hankel_matrix",
    "hosvd",
    "identity_basis",
    "mode_n_product",
    "outer2",
    "patch_tensor",
    "psnr",
    "random_orthonormal",
    "reconstruct_1d",
    "reconstruct_2d",
    "relative_error",
    "structure_add_1d",
    "structure_add_2d",
    "synthesize_1d",
    "synthesize_2d",
    "truncated_hosvd",
    "unfold",
]
