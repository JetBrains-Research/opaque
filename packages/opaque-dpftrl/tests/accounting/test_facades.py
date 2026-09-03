def test_accounting_mechanism_types_facade_imports():
    from opaque.dpftrl.accounting.mechanisms.types import MfGaussian

    assert MfGaussian.__name__ == "MfGaussian"
