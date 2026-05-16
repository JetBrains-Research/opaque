"""Per-model patch factory.

:func:`make_apply_model_patches` produces the ``apply_X_patches``
function for a HuggingFace model family.  It composes:

1. An ``apply_X_family_patches`` function (built by
   :func:`opaque.patches.transformers._family.make_apply_family_patches`)
   that mutates the family's modeling module on first encounter
   (idempotent, lazy).
2. Per-class patches on the *current model instance*: MLP, RMSNorm,
   DecoderLayer (for fused-add variants), and ForCausalLM.

This collapses what was previously ~17 hand-rolled per-model files into
~17 small factory invocations.  Architectural choices (which MLP kind,
which RMSNorm casting mode, whether the family supports fused-add-RMS)
are encoded in the factory call's kwargs and closed over the returned
function — so e.g. Gemma's ``activation_kind="geglu_exact"`` cannot accidentally
dispatch to SwiGLU.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Literal

from opaque.api.patches.transformers._router import _patch_forward
from opaque.api.patches.transformers.components.batchify import apply_batchify_patch
from opaque.api.patches.transformers.components.cross_entropy import (
    _make_fused_ce_causal_lm_forward,
    apply_causal_lm_loss_function_patch,
)
from opaque.api.patches.transformers.components.fused_add_rms_norm import (
    _fused_add_rms_fac_gemma,
    _fused_add_rms_fac_granite,
    _fused_add_rms_fac_llama,
    _fused_add_rms_fac_phi3,
)
from opaque.api.patches.transformers.components.geglu import (
    _make_geglu_approx_mlp_forward,
    _make_geglu_exact_mlp_forward,
)
from opaque.api.patches.transformers.components.kv_cache import apply_kv_cache_patch
from opaque.api.patches.transformers.components.rms_norm import (
    _rmsnorm_fac_gemma,
    _rmsnorm_fac_gemma2,
    _rmsnorm_fac_glm4,
    _rmsnorm_fac_llama,
    _rmsnorm_fac_olmo2,
)
from opaque.api.patches.transformers.components.swiglu import (
    _make_phi3_mlp_forward,
    _make_swiglu_mlp_forward,
)


log = logging.getLogger(__name__)


# Dispatch tables — single source of truth for which factory function
# implements each kind.  Adding a new activation / RMSNorm variant:
# register here, then reference by string from per-model factory calls.
ActivationKind = Literal["swiglu", "phi3_swiglu", "geglu_exact", "geglu_approx"]
RmsNormKind = Literal["llama", "gemma", "gemma2", "olmo2", "glm4"]
FusedAddRmsKind = Literal["llama", "gemma", "phi3", "granite"]

_ACTIVATION_FACTORIES = {
    "swiglu": _make_swiglu_mlp_forward,
    "phi3_swiglu": _make_phi3_mlp_forward,
    "geglu_exact": _make_geglu_exact_mlp_forward,
    "geglu_approx": _make_geglu_approx_mlp_forward,
}

_RMSNORM_FACTORIES = {
    "llama": _rmsnorm_fac_llama,
    "gemma": _rmsnorm_fac_gemma,
    "gemma2": _rmsnorm_fac_gemma2,
    "olmo2": _rmsnorm_fac_olmo2,
    "glm4": _rmsnorm_fac_glm4,
}

_FUSED_ADD_RMS_FACTORIES = {
    "llama": _fused_add_rms_fac_llama,
    "gemma": _fused_add_rms_fac_gemma,
    "phi3": _fused_add_rms_fac_phi3,
    "granite": _fused_add_rms_fac_granite,
}


# ----------------------------------------------------------------------------
# Public registration helpers — let users plug their own kernel variants in.
# ----------------------------------------------------------------------------


def register_activation_kind(
    name: str,
    factory: Callable,
) -> None:
    """Register a custom gated-activation forward factory under ``name``.

    Args:
        name: Identifier used in :func:`make_apply_model_patches` as
            ``activation_kind=name``.
        factory: Callable taking the original module's bound ``forward``
            and returning a new forward.
    """
    _ACTIVATION_FACTORIES[name] = factory


def register_rms_norm_kind(name: str, factory: Callable) -> None:
    """Register a custom RMSNorm forward factory under ``name`` (read by
    ``make_apply_model_patches(rms_norm_kind=name)``)."""
    _RMSNORM_FACTORIES[name] = factory


def register_fused_add_rms_kind(name: str, factory: Callable) -> None:
    """Register a custom fused-add-RMSNorm DecoderLayer factory under
    ``name`` (read by
    ``make_apply_model_patches(fused_add_rms_kind=name)``)."""
    _FUSED_ADD_RMS_FACTORIES[name] = factory


def _resolve(spec, registry: dict[str, Callable]) -> Callable | None:
    """Convert a kind-spec to a factory callable.

    ``spec`` accepts: ``None`` (no patch), a registered string name,
    or a callable used directly.
    """
    if spec is None:
        return None
    if callable(spec):
        return spec
    if spec in registry:
        return registry[spec]
    raise ValueError(
        f"Unknown kind {spec!r}; registered: {sorted(registry)}.  "
        "Pass a callable directly, or use ``register_*_kind`` to add a custom name."
    )


def make_apply_model_patches(
    *,
    family: str,
    family_apply: Callable,
    module_path: str,
    classes: dict[str, str],
    activation_kind: str | Callable | None = None,
    rms_norm_kind: str | Callable | None = None,
    fused_add_rms_kind: str | Callable | None = None,
) -> Callable:
    """Build an ``apply_X_patches`` function for a given family.

    The returned function applies family-runtime patches lazily on first
    call (via ``family_apply``) and patches per-class methods on the
    given model instance.  Architecture choices are closed over the
    returned function so e.g. an MLP kernel can't be cross-wired to the
    wrong activation flag.

    Args:
        family: Family name (matches :func:`family_name`).
        family_apply: The matching ``apply_X_family_patches`` produced
            by :func:`make_apply_family_patches`.  Called first inside
            the returned function (idempotent).
        module_path: Dotted path to the modeling module.
        classes: Mapping of role → HF class name.  Recognized roles:
            ``"mlp"``, ``"rms_norm"``, ``"decoder_layer"``,
            ``"causal_lm"``.  Roles absent from the mapping are skipped
            (e.g. Cohere has no RMSNorm; omit the ``"rms_norm"`` entry).
        activation_kind: Which gated-activation forward factory to use.
            Either a registered string name (``"swiglu"``,
            ``"geglu_exact"``, …), a callable used directly, or ``None``
            (no activation patch).  The user-facing kwarg that gates this
            concern is always ``activation``; the family decides which
            concrete activation kernel is deployed.
        rms_norm_kind: Same shape — registered name, callable, or ``None``.
        fused_add_rms_kind: Same shape — for the DecoderLayer fused-add
            variant.

    Returns:
        Callable with signature
        ``apply(model=None, *, performance=True, compat=True, **kwargs) -> None``.
        Per-concern kwargs default into the right bucket:
        ``rope``, ``rms_norm``, ``activation``, ``cross_entropy``,
        ``fused_linear_cross_entropy``, ``kv_cache`` → ``performance``;
        ``eager_attention``, ``batchify`` → ``compat``.
    """
    activation_factory = _resolve(activation_kind, _ACTIVATION_FACTORIES)
    rms_norm_factory = _resolve(rms_norm_kind, _RMSNORM_FACTORIES)
    fused_add_rms_factory = _resolve(fused_add_rms_kind, _FUSED_ADD_RMS_FACTORIES)

    def apply(
        model=None,
        *,
        performance: bool = True,
        compat: bool = True,
        **kwargs,
    ) -> None:
        # Lazily apply family-runtime patches.  Idempotent.
        family_apply(performance=performance, compat=compat, **kwargs)

        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            return

        # Gated activation inside the model's MLP module (SwiGLU / GeGLU /
        # Phi3-style SwiGLU).  ``performance`` remains the coarse gate;
        # ``activation`` is the fine-grained override.
        if activation_factory is not None and kwargs.get("activation", performance):
            mlp_class = classes.get("mlp")
            if mlp_class is not None:
                _patch_forward(getattr(mod, mlp_class, None), activation_factory, model)

        # RMSNorm (unified standalone + fused-add).
        rms_norm_on = kwargs.get("rms_norm", performance)
        if rms_norm_on:
            if rms_norm_factory is not None:
                rms_norm_class = classes.get("rms_norm")
                if rms_norm_class is not None:
                    _patch_forward(
                        getattr(mod, rms_norm_class, None),
                        rms_norm_factory,
                        model,
                    )
            if fused_add_rms_factory is not None:
                decoder_class = classes.get("decoder_layer")
                if decoder_class is not None:
                    _patch_forward(
                        getattr(mod, decoder_class, None),
                        fused_add_rms_factory,
                        model,
                    )

        # Cross-entropy patches split into two gates so callers needing
        # logits (e.g. SFTTrainer with ``compute_metrics`` /
        # ``preprocess_logits_for_metrics``) can disable the fused
        # linear+CE forward (which sets ``logits=None``) while keeping
        # the non-fused CE kernel on materialized logits.
        #   - ``cross_entropy``: ``loss_function`` → ``Opaque_CrossEntropyLoss``
        #     (kernel speedup on already-materialized logits; safe — returns logits).
        #   - ``fused_linear_cross_entropy``: ``forward`` → ``Opaque_LinearCrossEntropyLoss``
        #     (skips ``lm_head`` materialization for max memory savings; returns
        #     ``logits=None`` on the fused path).
        # ``fused_linear_cross_entropy`` cascades from ``cross_entropy`` so the
        # historical ``cross_entropy=False`` opt-out still turns both off.
        causal_lm_class = classes.get("causal_lm")
        causal_lm_obj = getattr(mod, causal_lm_class, None) if causal_lm_class else None
        cross_entropy_on = kwargs.get("cross_entropy", performance)
        if cross_entropy_on and causal_lm_obj is not None:
            apply_causal_lm_loss_function_patch(model, causal_lm_obj)
        if (
            kwargs.get("fused_linear_cross_entropy", cross_entropy_on)
            and causal_lm_obj is not None
        ):
            _patch_forward(
                causal_lm_obj,
                _make_fused_ce_causal_lm_forward,
                model,
            )

        # vmap-safety patch on the causal-LM class.
        if kwargs.get("batchify", compat) and causal_lm_obj is not None:
            apply_batchify_patch(causal_lm_obj, model)
        # KV cache disabler: avoids wasted DynamicCache allocation per
        # forward during training (and prevents vmap memory leaks from
        # the cache's circular refs). Performance-bucket since it's about
        # memory efficiency in the common training path.
        if kwargs.get("kv_cache", performance) and causal_lm_obj is not None:
            apply_kv_cache_patch(causal_lm_obj, model)

    apply.__name__ = f"apply_{family}_patches"
    apply.__qualname__ = apply.__name__
    apply._opaque_family = family  # type: ignore[attr-defined]
    return apply


__all__ = [
    "make_apply_model_patches",
    "register_activation_kind",
    "register_rms_norm_kind",
    "register_fused_add_rms_kind",
    "ActivationKind",
    "RmsNormKind",
    "FusedAddRmsKind",
]
