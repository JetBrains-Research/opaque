"""Column-normalized banded lower-triangular matrices for BandMF.

Implements the ColumnNormalizedBanded class, which represents banded
lower-triangular matrices where each column is normalized to have L2 norm 1.

References:
    - Algorithm 9: https://arxiv.org/abs/2306.08153
    - Scaling BandMF: https://arxiv.org/abs/2405.15913
"""

from __future__ import annotations

import dataclasses

import torch

from . import sensitivity, streaming_matrix


@dataclasses.dataclass
class ColumnNormalizedBanded:
    """A column-normalized banded lower-triangular n x n matrix.

    Parameterized by an n x b matrix ``params``. The matrix C is obtained
    by placing params in the first b bands and normalizing each column to
    have L2 norm 1.

    Example::

        params = [[a b c]    C = [a        ]
                  [d e f]        [b d      ]
                  [g h i]        [c e g    ]
                  [j k -]        [  f h j  ]
                  [m - -]]       [    i k m]

    Attributes:
        params: An n x b tensor of banded parameters.
    """

    params: torch.Tensor

    @property
    def n(self) -> int:
        return self.params.shape[0]

    @property
    def bands(self) -> int:
        return self.params.shape[1]

    @classmethod
    def from_banded_toeplitz(
        cls, n: int, coefs: torch.Tensor
    ) -> ColumnNormalizedBanded:
        """Construct from banded Toeplitz coefficients.

        Args:
            n: Number of training iterations.
            coefs: Array of b Toeplitz coefficients.

        Returns:
            A ColumnNormalizedBanded representation.
        """
        bands = len(coefs)
        if bands > n or bands < 1:
            raise ValueError(f"len(coefs) must be in [1, n], got {bands}")
        coefs = coefs / torch.linalg.norm(coefs)
        params = coefs.unsqueeze(0).expand(n, -1).clone()
        # Set the lower-right triangle to 0
        params_rev = params.flip(0)
        mask = torch.tril(torch.ones(n, bands, dtype=params.dtype))
        params = (params_rev * mask).flip(0)
        return cls(params=params)

    @classmethod
    def default(cls, n: int, bands: int) -> ColumnNormalizedBanded:
        """Construct using Fichtenberger et al. initialization.

        Args:
            n: Number of iterations.
            bands: Number of bands.

        Returns:
            A ColumnNormalizedBanded with default initialization.
        """
        k = torch.arange(bands, dtype=torch.float64)
        coefs = torch.cumprod(
            ((2 * k - 1) / (2 * k)).clamp(min=0).clone().detach(), dim=0
        )
        coefs[0] = 1.0
        return cls.from_banded_toeplitz(n, coefs)

    def materialize(self) -> torch.Tensor:
        """Convert to a dense n x n matrix.

        Returns:
            Dense lower-triangular column-normalized matrix.
        """
        row_idx = torch.arange(self.n, dtype=torch.long).unsqueeze(1)
        col_idx = torch.arange(self.n, dtype=torch.long).unsqueeze(0)
        D = row_idx - col_idx
        # Index into flattened params
        indexer = (D + self.bands * col_idx + 1) * (D >= 0) * (D < self.bands)
        flat_params = torch.cat(
            [torch.zeros(1, dtype=self.params.dtype), self.params.flatten()]
        )
        C = flat_params[indexer.long()]
        # Column normalize
        col_norms = torch.linalg.norm(C, dim=0)
        col_norms = col_norms.clamp(min=1e-12)  # Avoid division by zero
        return C / col_norms

    def inverse_as_streaming_matrix(
        self,
    ) -> streaming_matrix.StreamingMatrix:
        """Create C^{-1} as a StreamingMatrix.

        Implements Algorithm 9 from https://arxiv.org/abs/2306.08153.

        Returns:
            StreamingMatrix representing C^{-1}.
        """
        params = self.params
        n = self.n
        b = self.bands

        def init_fn(abstract_value):
            dtype = torch.promote_types(abstract_value.dtype, params.dtype)
            zero = torch.zeros_like(abstract_value, dtype=dtype)
            buffers = zero.unsqueeze(0).expand(b, *zero.shape).clone()
            return (torch.tensor(0, dtype=torch.long), buffers)

        def next_fn(value, state):
            index, bufs = state
            if b == 1:
                return value, (index + 1, bufs)

            k = int(index.item()) % b
            r = torch.arange(b, dtype=torch.long)

            # Get the row of params for this index
            idx = index.item()
            if idx >= n:
                # Beyond the matrix, identity behavior
                return value, (index + 1, bufs)

            row = torch.zeros(b, dtype=params.dtype)
            for j in range(b):
                src_idx = idx - int(r[j].item())
                if 0 <= src_idx < n:
                    row[j] = params[src_idx, int(r[j].item())]

            # Algorithm 9: xi = (value - row[1:] @ bufs[k-r][1:]) / row[0]
            buf_indices = [(k - int(r[j].item())) % b for j in range(1, b)]
            selected_bufs = torch.stack([bufs[bi] for bi in buf_indices])
            inner = torch.tensordot(row[1:], selected_bufs, dims=1)
            xi = (value - inner) / row[0].clamp(min=1e-15)

            col_norm = torch.linalg.norm(params[idx])
            new_bufs = bufs.clone()
            new_bufs[k] = xi
            return xi * col_norm, (index + 1, new_bufs)

        return streaming_matrix.StreamingMatrix.from_array_implementation(
            init_fn, next_fn
        )


def minsep_sensitivity_squared(
    strategy: ColumnNormalizedBanded,
    min_sep: int,
    max_participations: int | None = None,
    n: int | None = None,
    skip_checks: bool = False,
) -> int:
    """Returns the sensitivity squared of a ColumnNormalizedBanded strategy.

    For column-normalized banded matrices with min_sep >= bands,
    sensitivity = max_participations.

    Args:
        strategy: The strategy matrix.
        min_sep: Minimum separation between participations.
        max_participations: Maximum participations.
        n: Optional matrix size.
        skip_checks: Skip input validation.

    Returns:
        The sensitivity squared (= max_participations for normalized banded).
    """
    bands = strategy.bands
    n = n or strategy.n
    max_participations = sensitivity.minsep_true_max_participations(
        n, min_sep, max_participations
    )
    if not skip_checks:
        if min_sep < bands:
            raise ValueError(
                f"min_sep={min_sep} must be >= bands={bands}. "
                "This usually indicates a mis-configuration."
            )
        if n > strategy.n:
            raise ValueError(f"n={n} must be <= strategy.n={strategy.n}.")
    return max_participations


def per_query_error(
    strategy: ColumnNormalizedBanded,
    workload: streaming_matrix.StreamingMatrix | None = None,
) -> torch.Tensor:
    """Compute expected per-query squared error.

    Args:
        strategy: The strategy matrix (C).
        workload: The workload matrix (defaults to prefix sum).

    Returns:
        Per-query expected squared error, tensor of length n.
    """
    if workload is None:
        workload = streaming_matrix.prefix_sum()
    B = workload @ strategy.inverse_as_streaming_matrix()
    return B.row_norms_squared(strategy.n)


def mean_error(*args, **kwargs):
    """Mean per-query error."""
    return per_query_error(*args, **kwargs).mean()


def max_error(*args, **kwargs):
    """Max per-query error."""
    return per_query_error(*args, **kwargs).max()
