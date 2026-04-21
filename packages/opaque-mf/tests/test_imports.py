"""Smoke test: opaque.mf namespace root and key leaves import cleanly.

In particular, ``import opaque.mf`` must not require the Rust-backed
``opaque.accounting`` native module (it is lazy-loaded inside calibration
helpers), so that a fresh ``pip install opaque-mf`` boots on a system
without the accounting extension.
"""

import sys


def test_mf_package_imports():
    import opaque.mf as mf

    assert mf.__version__


def test_mf_leaves_importable():
    from opaque.mf.noise import (  # noqa: F401
        band_mf_strategy,
        bisr_strategy,
        blt_strategy,
        bsr_strategy,
        identity_strategy,
        jme_noise,
        lambda_cgd_strategy,
        mf_noise,
    )
    from opaque.mf.optimizers.adamw_jme import adamw_jme  # noqa: F401
    from opaque.mf.sampling import (  # noqa: F401
        BallsInBinsSampler,
        BMinSepSampler,
        CyclicPoissonSampler,
        SequentialBatchSampler,
    )


def test_mf_import_does_not_load_native_accounting():
    """``import opaque.mf`` must not pull the accounting native extension."""
    # Force a clean state for this assertion
    for m in [
        m
        for m in sys.modules
        if m.startswith("opaque.mf") or m.startswith("opaque.accounting")
    ]:
        del sys.modules[m]
    import opaque.mf  # noqa: F401

    assert "opaque.accounting" not in sys.modules, (
        "opaque.mf top-level import must lazy-load opaque.accounting"
    )
    assert "opaque.accounting._native" not in sys.modules, (
        "opaque.mf top-level import must not load the accounting native extension"
    )
