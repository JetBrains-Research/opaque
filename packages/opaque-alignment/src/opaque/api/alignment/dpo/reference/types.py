"""Reference-model configuration record for the alignment ``reference/`` concern.

Defines :class:`RefSpec` — the inert, frozen record a trainer or example uses to
declare *which* of the four reference-model configurations (plan §7.8) it
expects. The four configs are dispatched at runtime by ``null_ref_context`` and
``compute_ref_logprobs_for_dataset`` based on the actual model/ref inputs; the
``RefSpec`` is the caller's *declaration* of intent, not a dispatcher itself.

The four ``kind`` values map onto the plan §7.8 design table:

==========================  ==================================================
``kind``                    Plan §7.8 row
==========================  ==================================================
``"separate_model"``        Separate ``ref_model`` (not PEFT-derived). The
                            reference is an independent model; no adapter to
                            toggle.
``"lora_ref_adapter"``      LoRA model carrying a named ``"ref"`` adapter
                            clone. ``adapter_name`` records which adapter to
                            activate as the reference.
``"lora_disable_adapter"``  LoRA model with no separate ref; the reference is
                            the base model reached by disabling the adapter
                            (TR-DPO seed / ad-hoc).
``"callable"``              An explicit user-supplied reference function over
                            tokens (advanced). No adapter to toggle.
==========================  ==================================================

This is pure metadata: no tensors, no torch import. AGENTS.md rule 9 permits
frozen dataclasses for inert state of this kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["RefSpec"]


RefKind = Literal[
    "separate_model",
    "lora_ref_adapter",
    "lora_disable_adapter",
    "callable",
]


@dataclass(frozen=True)
class RefSpec:
    """Declares which reference-model configuration the caller expects (§7.8).

    An inert record read by trainers and the precompute / ``null_ref_context``
    helpers. It carries no behaviour; the helpers dispatch on the actual
    model/ref inputs, using this record only to declare and validate the
    intended configuration.

    Attributes:
        kind: One of the four reference configurations from the plan §7.8
            table — ``"separate_model"``, ``"lora_ref_adapter"``,
            ``"lora_disable_adapter"``, or ``"callable"``.
        adapter_name: For ``"lora_ref_adapter"``, the name of the PEFT adapter
            to activate as the reference (typically ``"ref"``). ``None`` for
            the other three configurations, where no named adapter is selected.
    """

    kind: RefKind
    adapter_name: str | None = None
