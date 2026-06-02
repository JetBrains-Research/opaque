"""opaque.alignment — functional primitives for DP-safe preference learning.

Method-first layout, mirroring ``opaque.dpsgd`` / ``opaque.dpftrl``: each
method owns its primitives under its own namespace —

- :mod:`opaque.alignment.dpo` — DPO loss family, preference collator,
  fused-linear preference kernel, reference-model helpers, reward telemetry,
  and preference prompt extraction.
- :mod:`opaque.alignment.sft` — NLL / DFT losses and the language-modeling
  collator.
- :mod:`opaque.alignment.data` — shared, method-agnostic chat-template data
  prep: install a training chat template, then tokenize chat turns into
  ``input_ids`` + a ``completion_mask`` for completion-only loss.

Other shared, lower-level primitives (logprob, general token metrics) are
internal impl under ``opaque.api.alignment.*`` and are surfaced through the
method that consumes them (e.g. ``sequence_logp`` via
:mod:`opaque.alignment.dpo`), following the shared-impl re-import pattern of
``opaque.dpsgd.clipping``.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from opaque.alignment import data, dpo, sft

try:
    __version__ = _pkg_version("opaque-alignment")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__", "data", "dpo", "sft"]
