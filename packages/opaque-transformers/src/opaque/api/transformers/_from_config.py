"""Single dispatch entry point: upstream HF / TRL config → opaque equivalent.

:func:`from_hf_config` is the one public converter users reach for — it
type-dispatches an upstream config to the matching opaque config class:

- ``trl.DPOConfig``               → :class:`opaque...trl.DPOConfig`
- ``trl.SFTConfig``               → :class:`opaque...trl.SFTConfig`
- ``transformers.TrainingArguments`` → :class:`opaque...trainer.TrainingArguments`

It's a thin wrapper over the per-class converters
(:meth:`DPOConfig.from_trl`, :meth:`SFTConfig.from_trl`,
:meth:`TrainingArguments.from_hf`); use those directly if you already know the
input type.
"""

from __future__ import annotations

from typing import Any


def from_hf_config(config: Any, *, strict: bool = True, **overrides: Any) -> Any:
    """Convert an upstream HF / TRL config to its opaque equivalent.

    Dispatches on the runtime type of ``config``. TRL's ``SFTConfig`` /
    ``DPOConfig`` subclass HF ``TrainingArguments``, so they are matched first.

    A DP knob is required, exactly as on the underlying converters: pass either
    ``privacy_noise_multiplier=<float>`` or ``privacy_target_epsilon=<float>``.
    Any other keyword (e.g. ``clipping_norm``, ``use_performance_kernels``,
    ``microbatch_size``) overrides the converted field by name after
    translation.

    Parameters
    ----------
    config
        A ``transformers.TrainingArguments``, ``trl.SFTConfig``, or
        ``trl.DPOConfig`` instance.
    strict
        Forwarded to the converter: when ``True``, emit ``RuntimeWarning`` for
        dropped non-default fields.
    **overrides
        DP knobs + any opaque field to override by name.

    Returns
    -------
    The matching opaque config instance.

    Raises
    ------
    TypeError
        If ``config`` is not a recognized HF / TRL config type.
    ValueError
        If no DP knob is supplied, or a field has no opaque equivalent.
    """
    # TRL configs subclass HF ``TrainingArguments``, so match them first; a
    # missing ``trl`` extra just means the input can't be a TRL config.
    try:
        import trl
    except ImportError:
        trl = None

    if trl is not None:
        if isinstance(config, trl.DPOConfig):
            from .trl import DPOConfig

            return DPOConfig.from_trl(config, strict=strict, **overrides)
        if isinstance(config, trl.SFTConfig):
            from .trl import SFTConfig

            return SFTConfig.from_trl(config, strict=strict, **overrides)

    try:
        from transformers import TrainingArguments as HFTrainingArguments
    except ImportError:
        HFTrainingArguments = None

    if HFTrainingArguments is not None and isinstance(config, HFTrainingArguments):
        from .trainer import TrainingArguments

        return TrainingArguments.from_hf(config, strict=strict, **overrides)

    raise TypeError(
        f"from_hf_config expects a ``transformers.TrainingArguments``, "
        f"``trl.SFTConfig``, or ``trl.DPOConfig`` instance, got "
        f"{type(config).__name__}. (If you meant a TRL config, install the "
        f"optional ``opaque[trl]`` extra.)"
    )
