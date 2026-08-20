"""Public DP-SGD façade tests."""


def test_accounting_facade_is_lazy_loaded(monkeypatch):
    import opaque.dpsgd as dpsgd

    monkeypatch.delitem(dpsgd.__dict__, "accounting", raising=False)

    accounting = dpsgd.accounting

    assert accounting.__name__ == "opaque.dpsgd.accounting"
