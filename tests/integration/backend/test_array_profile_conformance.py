"""Cross-provider conformance for the optional array-operation profiles.

The profiles back the matrix-factorization strategy math, which needs matrix
construction, cumulative scans, dense linear algebra, and real FFTs. Values are
compared against NumPy references rather than across providers, matching the
keyed-randomness rule in ``docs/development/backend-providers.md``.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.integration.backend._providers import provider_case

from opaque import ops
from opaque.api.engine.backend import clear_backend, use_backend

_PROVIDERS = ["torch", "jax", "mlx"]


@pytest.fixture(autouse=True)
def _reset_backend() -> None:
    clear_backend()
    yield
    clear_backend()


def _high_precision(case, values):
    """Build an array at the provider's best accumulation precision."""
    seed = case.array(values, None)
    return ops.astype(seed, ops.accumulator_dtype(seed))


def _tolerance(value) -> float:
    """Derive a tolerance from the precision the provider actually delivers.

    MLX has no float64, and JAX yields float32 unless ``jax_enable_x64`` is
    set, so a fixed per-provider table would encode today's configuration
    rather than the precision in play.
    """
    return float(ops.finfo_eps(ops.dtype(value))) * 100.0


@pytest.mark.parametrize("provider_name", _PROVIDERS)
def test_every_array_profile_is_supported(provider_name: str) -> None:
    case = provider_case(provider_name)
    for profile in ops.ArrayProfile:
        assert profile.supports(case.backend), (
            f"{provider_name} does not implement the {profile.value!r} array "
            "profile; MF strategy math cannot run on it."
        )


@pytest.mark.parametrize("provider_name", _PROVIDERS)
def test_construction_and_layout_match_numpy(provider_name: str) -> None:
    case = provider_case(provider_name)
    with use_backend(case.backend):
        x = _high_precision(case, [1.0, 3.0, 2.0])
        atol = _tolerance(x)

        np.testing.assert_allclose(
            case.to_numpy(ops.flip(x, 0)), [2.0, 3.0, 1.0], atol=atol
        )
        np.testing.assert_allclose(
            case.to_numpy(ops.roll(x, 1, 0)), [2.0, 1.0, 3.0], atol=atol
        )
        np.testing.assert_allclose(
            case.to_numpy(ops.tril(ops.ones((3, 3), like=x))),
            np.tril(np.ones((3, 3))),
            atol=atol,
        )
        np.testing.assert_allclose(
            case.to_numpy(ops.diag(x)), np.diag([1.0, 3.0, 2.0]), atol=atol
        )
        np.testing.assert_allclose(
            case.to_numpy(ops.eye(3, like=x)), np.eye(3), atol=atol
        )
        np.testing.assert_allclose(
            case.to_numpy(ops.stack([x, x], axis=0)),
            np.stack([[1.0, 3.0, 2.0]] * 2),
            atol=atol,
        )

        grid = ops.reshape(ops.arange(0, 6, 1, like=x), (2, 3))
        assert ops.shape(ops.transpose(grid)) == (3, 2)
        np.testing.assert_allclose(
            case.to_numpy(ops.transpose(grid)), np.arange(6).reshape(2, 3).T, atol=atol
        )

        # ``asarray`` is the host -> device seam the strategy math relies on.
        np.testing.assert_allclose(
            case.to_numpy(ops.asarray(np.array([4.0, 5.0]), like=x)),
            [4.0, 5.0],
            atol=atol,
        )


@pytest.mark.parametrize("provider_name", _PROVIDERS)
def test_scans_and_ordering_match_numpy(provider_name: str) -> None:
    case = provider_case(provider_name)
    with use_backend(case.backend):
        x = _high_precision(case, [1.0, 3.0, 2.0])
        atol = _tolerance(x)

        np.testing.assert_allclose(
            case.to_numpy(ops.cumsum(x)), [1.0, 4.0, 6.0], atol=atol
        )
        np.testing.assert_allclose(
            case.to_numpy(ops.cumprod(x)), [1.0, 3.0, 6.0], atol=atol
        )
        np.testing.assert_allclose(
            case.to_numpy(ops.cummax(x)), [1.0, 3.0, 3.0], atol=atol
        )
        assert float(ops.scalar_item(ops.prod(x))) == pytest.approx(6.0, abs=atol)
        assert float(ops.scalar_item(ops.amax(x))) == pytest.approx(3.0, abs=atol)
        assert float(ops.scalar_item(ops.amin(x))) == pytest.approx(1.0, abs=atol)
        assert int(ops.scalar_item(ops.argmax(x))) == 1
        np.testing.assert_array_equal(case.to_numpy(ops.argsort(x)), [0, 2, 1])
        np.testing.assert_array_equal(
            case.to_numpy(ops.argsort(x, descending=True)), [1, 2, 0]
        )
        np.testing.assert_array_equal(
            case.to_numpy(ops.nonzero(_high_precision(case, [0.0, 1.0, 0.0, 2.0]))),
            [1, 3],
        )
        assert bool(ops.scalar_item(ops.any(ops.greater(x, 2.5))))
        assert not bool(ops.scalar_item(ops.any(ops.greater(x, 9.0))))


@pytest.mark.parametrize("provider_name", _PROVIDERS)
def test_linalg_and_spectral_match_numpy(provider_name: str) -> None:
    case = provider_case(provider_name)
    with use_backend(case.backend):
        x = _high_precision(case, [1.0, 3.0, 2.0])
        atol = _tolerance(x)
        m = _high_precision(case, [[2.0, 1.0], [0.0, 3.0]])

        np.testing.assert_allclose(
            case.to_numpy(ops.matmul(m, m)),
            np.array([[4.0, 5.0], [0.0, 9.0]]),
            atol=atol,
        )
        assert float(ops.scalar_item(ops.tensordot(x, x, 1))) == pytest.approx(
            14.0, abs=atol
        )
        np.testing.assert_allclose(
            case.to_numpy(ops.linalg_inv(m)),
            np.linalg.inv([[2.0, 1.0], [0.0, 3.0]]),
            atol=atol,
        )
        # Upper-triangular, so the eigenvalues are the diagonal. Sorted because
        # providers do not agree on eigenvalue order.
        eigenvalues = np.sort(case.to_numpy(ops.real(ops.linalg_eigvals(m))))
        np.testing.assert_allclose(eigenvalues, [2.0, 3.0], atol=atol)

        # Real FFT round-trip recovers the input.
        np.testing.assert_allclose(
            case.to_numpy(ops.fft_irfft(ops.fft_rfft(x, n=3), n=3)),
            [1.0, 3.0, 2.0],
            atol=atol,
        )


@pytest.mark.parametrize("provider_name", _PROVIDERS)
def test_fft_convolution_matches_direct_convolution(provider_name: str) -> None:
    """The FFT path is how Toeplitz coefficient products are computed."""
    case = provider_case(provider_name)
    left_np = np.array([1.0, 0.5, 0.25])
    right_np = np.array([1.0, -0.5, 0.125])
    with use_backend(case.backend):
        left = _high_precision(case, left_np.tolist())
        right = _high_precision(case, right_np.tolist())
        atol = _tolerance(left)
        full_len = len(left_np) + len(right_np) - 1
        product = ops.multiply(
            ops.fft_rfft(left, n=full_len), ops.fft_rfft(right, n=full_len)
        )
        conv = case.to_numpy(ops.fft_irfft(product, n=full_len))

    np.testing.assert_allclose(conv, np.convolve(left_np, right_np), atol=atol)
