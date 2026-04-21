"""Smoke test: opaque.core namespace and key leaf modules import cleanly."""


def test_namespace_root_imports():
    import opaque.core

    assert opaque.core.__version__
    assert hasattr(opaque.core, "clipping")
    assert hasattr(opaque.core, "sampling")
    assert hasattr(opaque.core, "random")
    assert hasattr(opaque.core, "utils")
    assert hasattr(opaque.core, "distributed")
    assert hasattr(opaque.core, "profiling")
    assert hasattr(opaque.core, "noise")


def test_leaf_modules_import():
    from opaque.core.clipping import clip_pytree, clipped_grad  # noqa: F401
    from opaque.core.noise.types import NoiseState  # noqa: F401
    from opaque.core.random import RngKey, key  # noqa: F401
    from opaque.core.sampling import PoissonSampler, poisson_collate  # noqa: F401
    from opaque.core.utils import PerGroup, tree_map  # noqa: F401
