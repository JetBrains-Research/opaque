"""``opaque.device`` resolves every name it ever exported, and says what moved.

The alias exists only to carry callers to :mod:`opaque.torch.device`.  Its
resolution is dynamic, so a wrong ``__all__`` or ``_RELOCATED`` entry would
import cleanly and only fail at a user's first access — these tests are what
catch that instead.
"""

from __future__ import annotations

import importlib
import warnings

import pytest

import opaque.device as alias
import opaque.torch.device as canonical
import opaque.torch.device.types as canonical_types

_CANONICAL = {
    "DeviceCapabilities": canonical_types,
    "device_capabilities": canonical,
    "fused_kernels_available": canonical,
    "sdpa_autocast_under_vmap_broken": canonical,
}


def _uncache(module):
    """Drop the names ``__getattr__`` memoised, so the next access warns again.

    ``importlib.reload`` re-executes the body over the *existing* module dict,
    so the cached entries survive it — they have to be removed by hand.
    """
    for name in module.__all__:
        module.__dict__.pop(name, None)
    return module


@pytest.fixture
def fresh_alias():
    """Give each test a shim that has not yet warned for any name."""
    yield _uncache(importlib.reload(alias))
    _uncache(alias)


def test_every_exported_name_is_the_canonical_object(fresh_alias) -> None:
    assert set(fresh_alias.__all__) == set(_CANONICAL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for name, home in _CANONICAL.items():
            assert getattr(fresh_alias, name) is getattr(home, name)


@pytest.mark.parametrize("name", sorted(_CANONICAL))
def test_access_names_the_replacement(fresh_alias, name: str) -> None:
    home = _CANONICAL[name].__name__
    with pytest.warns(DeprecationWarning, match=f"{home}.{name}"):
        getattr(fresh_alias, name)


def test_warns_once_per_name_not_once_per_lookup(fresh_alias) -> None:
    """``from opaque.device import x`` probes the attribute more than once."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        for _ in range(3):
            _ = fresh_alias.device_capabilities
    assert len(caught) == 1


def test_unknown_name_still_raises_attribute_error(fresh_alias) -> None:
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = fresh_alias.nope


def test_dir_lists_the_deprecated_surface(fresh_alias) -> None:
    assert dir(fresh_alias) == sorted(_CANONICAL)
