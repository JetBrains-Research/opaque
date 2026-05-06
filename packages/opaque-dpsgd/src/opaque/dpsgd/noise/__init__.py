"""DP-SGD noise mechanisms: Gaussian, truncated Gaussian, per-group."""

from opaque.dpsgd.noise.gaussian import GaussianNoiseState, gaussian_noise
from opaque.dpsgd.noise.per_group_noise import per_group_noise_stddev
from opaque.dpsgd.noise.truncated_gaussian import truncated_gaussian_noise

# Side-effect import: registers the GaussianNoiseState sync handler
# with opaque.distributed.sync().
from opaque.dpsgd.noise import distributed as _distributed  # noqa: F401

__all__ = [
    "gaussian_noise",
    "GaussianNoiseState",
    "truncated_gaussian_noise",
    "per_group_noise_stddev",
]
