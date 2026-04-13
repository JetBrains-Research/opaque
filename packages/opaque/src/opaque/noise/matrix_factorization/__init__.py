"""Matrix Factorization mechanisms for correlated noise in DP-SGD.

This module implements matrix factorization approaches for differentially
private machine learning, including:

- StreamingMatrix: Memory-efficient interface for lower-triangular matrix
  multiplication
- Sensitivity computation under various participation patterns
- Toeplitz matrix mechanisms (BandMF)
- Buffered Linear Toeplitz (BLT) mechanisms

These mechanisms enable correlated noise addition (vs independent noise in
standard DP-SGD), achieving 10-50% utility improvement at the same privacy
budget.

References:
    - BandMF: https://arxiv.org/abs/2306.08153
    - BLT: https://arxiv.org/abs/2404.16706
    - Multi-epoch BLT: https://arxiv.org/abs/2408.08868
"""

from opaque.noise.matrix_factorization.band_mf_noise import (
    BandMfStrategy,
    band_mf_strategy,
)
from opaque.noise.matrix_factorization.bisr_noise import BisrStrategy, bisr_strategy
from opaque.noise.matrix_factorization.blt_mf_noise import BltStrategy, blt_strategy
from opaque.noise.matrix_factorization.identity_mf_noise import (
    IdentityStrategy,
    identity_strategy,
)
from opaque.noise.matrix_factorization.lambda_cgd_noise import (
    LambdaCgdStrategy,
    lambda_cgd_strategy,
)
from opaque.noise.matrix_factorization.noise import MFNoiseState
from opaque.noise.matrix_factorization.streaming_matrix import (
    StreamingMatrix,
    diagonal,
    identity,
    momentum_sgd_matrix,
    multiply_array,
    multiply_streaming_matrices,
    prefix_sum,
    scale_rows_and_columns,
)

__all__ = [
    "BandMfStrategy",
    "BisrStrategy",
    "BltStrategy",
    "IdentityStrategy",
    "LambdaCgdStrategy",
    "MFNoiseState",
    "StreamingMatrix",
    "band_mf_strategy",
    "bisr_strategy",
    "blt_strategy",
    "diagonal",
    "identity",
    "identity_strategy",
    "lambda_cgd_strategy",
    "momentum_sgd_matrix",
    "multiply_array",
    "multiply_streaming_matrices",
    "prefix_sum",
    "scale_rows_and_columns",
]
