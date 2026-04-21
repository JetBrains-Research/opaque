"""Smoke test: opaque.auditing namespace root and key leaves import cleanly."""


def test_auditing_package_imports():
    import opaque.auditing as auditing

    assert auditing.__version__
    assert auditing.coin_flip is not None
    assert auditing.one_run is not None
    assert auditing.loss_scores is not None


def test_auditing_leaves_importable():
    from opaque.auditing import CoinFlip, OneRunEstimate, coin_flip, loss_scores, one_run  # noqa: F401
    from opaque.auditing.attacks.loss import loss_scores as _ls  # noqa: F401
    from opaque.auditing.one_run.estimate import OneRunEstimate as _ore  # noqa: F401
