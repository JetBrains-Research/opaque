"""Smoke test: opaque.dpsgd namespace root and key leaves import cleanly."""


def test_dpsgd_package_imports():
    import opaque.dpsgd as dpsgd

    assert dpsgd.__version__


def test_dpsgd_leaves_importable():
    from opaque.dpsgd.clipping import adaptive_clipped_grad, auto_clipped_grad  # noqa: F401
    from opaque.dpsgd.clipping.adaptive import AdaptiveClipState  # noqa: F401
    from opaque.dpsgd.noise import (  # noqa: F401
        gaussian_noise,
        per_group_noise_stddev,
        truncated_gaussian_noise,
    )
    from opaque.dpsgd.sampling import TruncatedPoissonSampler  # noqa: F401


def test_adaptive_sync_registers_on_import():
    """Importing dpsgd must register AdaptiveClipState sync in the core registry."""
    import opaque.dpsgd  # noqa: F401
    from opaque.core.distributed import _SYNC_REGISTRY
    from opaque.dpsgd.clipping.adaptive import AdaptiveClipState

    assert AdaptiveClipState in _SYNC_REGISTRY
