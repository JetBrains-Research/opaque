"""Matrix Factorization mechanisms for correlated noise in DP-SGD.

This module implements matrix factorization approaches for differentially
private machine learning, including:

- StreamingMatrix: Memory-efficient interface for lower-triangular matrix
  multiplication
- Sensitivity computation under various participation patterns
- Toeplitz matrix mechanisms (BandMF)

These mechanisms enable correlated noise addition (vs independent noise in
standard DP-SGD), achieving 10-50% utility improvement at the same privacy
budget.

References:
    - BandMF: https://arxiv.org/abs/2306.08153
"""

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
    "MFNoiseState",
    "StreamingMatrix",
    "diagonal",
    "identity",
    "momentum_sgd_matrix",
    "multiply_array",
    "multiply_streaming_matrices",
    "prefix_sum",
    "scale_rows_and_columns",
]
