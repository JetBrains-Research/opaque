"""Legacy re-export of Matrix Factorization mechanisms.

The canonical API is now ``opaque.noise.mf``. This module re-exports
for backward compatibility.
"""

from opaque.noise.mf import (
    BandMfStrategy,
    BisrStrategy,
    BltStrategy,
    IdentityStrategy,
    LambdaCgdStrategy,
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    identity_strategy,
    lambda_cgd_strategy,
)
from opaque.noise.mf._engine import MFNoiseState
from opaque.noise.mf._streaming_matrix import (
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
