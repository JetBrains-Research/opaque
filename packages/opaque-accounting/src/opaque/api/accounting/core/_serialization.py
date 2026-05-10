"""Serialization registration hook for :mod:`opaque.accounting`.

Import this module (as :mod:`opaque.accounting` does) so
:class:`~opaque.accounting._accountant.Accountant` is registered with
:mod:`opaque.serialization`.

:class:`~opaque.accounting._base.DpProcess` subclasses register in
``__init_subclass__`` when their defining module loads.
"""

from __future__ import annotations

from opaque.api.accounting.core._accountant import _register_accountant_serialization

_register_accountant_serialization()

__all__: list[str] = []
