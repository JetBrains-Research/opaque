"""DP-FTRL accounting mechanism factories façade (matrix factorization).

Single factory :func:`mf_gaussian(nm, strategy)` builds the accounting
mechanism for every MF strategy from :mod:`opaque.dpftrl.noise`.
"""

from opaque.api.accounting.dpftrl.mechanisms import mf_gaussian
from opaque.dpftrl.accounting.mechanisms import types

__all__ = ["mf_gaussian", "types"]
