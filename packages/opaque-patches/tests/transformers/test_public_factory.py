"""Public surface for downstream users to register their own model families.

Validates:

- Built-in kinds work via ``make_apply_model_patches`` (smoke).
- Custom kinds register, then resolve by name.
- Raw callables are accepted directly (no registration needed).
- Unknown string kinds raise a useful error listing the registered options.
- ``family_name`` is exposed at the package level and detects HF families.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("transformers")

from opaque.patches.transformers import (
    family_name,
    make_apply_family_patches,
    make_apply_model_patches,
    register_activation_kind,
    register_family,
    register_fused_add_rms_kind,
    register_rms_norm_kind,
    supported_families,
)
from opaque.api.patches.transformers._registry import _FAMILY_REGISTRY


@pytest.fixture
def _restore_registry():
    """Snapshot/restore the family registry around tests that mutate it."""
    snapshot = dict(_FAMILY_REGISTRY)
    yield
    _FAMILY_REGISTRY.clear()
    _FAMILY_REGISTRY.update(snapshot)


def _passthrough_factory(orig):
    """Trivial factory used as a stand-in for a real Triton kernel patch."""
    return orig


# ----------------------------------------------------------------------------
# make_apply_family_patches / make_apply_model_patches
# ----------------------------------------------------------------------------


def test_built_in_kinds_compose_into_a_callable():
    fam = make_apply_family_patches(
        family="public_api_test_a",
        module_path="transformers.models.llama.modeling_llama",
    )
    apply = make_apply_model_patches(
        family="public_api_test_a",
        family_apply=fam,
        module_path="transformers.models.llama.modeling_llama",
        classes={
            "mlp": "LlamaMLP",
            "rms_norm": "LlamaRMSNorm",
            "decoder_layer": "LlamaDecoderLayer",
            "causal_lm": "LlamaForCausalLM",
        },
        activation_kind="swiglu",
        rms_norm_kind="llama",
        fused_add_rms_kind="llama",
    )
    assert callable(apply)
    assert apply.__name__ == "apply_public_api_test_a_patches"
    assert apply._opaque_family == "public_api_test_a"


def test_unknown_kind_raises_with_registered_set_in_message():
    fam = make_apply_family_patches(
        family="public_api_test_unknown",
        module_path="transformers.models.llama.modeling_llama",
    )
    with pytest.raises(ValueError, match="Unknown kind 'nope'"):
        make_apply_model_patches(
            family="public_api_test_unknown",
            family_apply=fam,
            module_path="transformers.models.llama.modeling_llama",
            classes={},
            activation_kind="nope",
        )


# ----------------------------------------------------------------------------
# Custom registration
# ----------------------------------------------------------------------------


def test_register_activation_kind_then_use_by_name():
    register_activation_kind("public_api_glu", _passthrough_factory)
    fam = make_apply_family_patches(
        family="public_api_test_b",
        module_path="transformers.models.llama.modeling_llama",
    )
    apply = make_apply_model_patches(
        family="public_api_test_b",
        family_apply=fam,
        module_path="transformers.models.llama.modeling_llama",
        classes={"mlp": "LlamaMLP"},
        activation_kind="public_api_glu",
    )
    assert callable(apply)


def test_register_rms_norm_kind_then_use_by_name():
    register_rms_norm_kind("public_api_norm", _passthrough_factory)
    fam = make_apply_family_patches(
        family="public_api_test_c",
        module_path="transformers.models.llama.modeling_llama",
    )
    apply = make_apply_model_patches(
        family="public_api_test_c",
        family_apply=fam,
        module_path="transformers.models.llama.modeling_llama",
        classes={"rms_norm": "LlamaRMSNorm"},
        rms_norm_kind="public_api_norm",
    )
    assert callable(apply)


def test_register_fused_add_rms_kind_then_use_by_name():
    register_fused_add_rms_kind("public_api_fused", _passthrough_factory)
    fam = make_apply_family_patches(
        family="public_api_test_d",
        module_path="transformers.models.llama.modeling_llama",
    )
    apply = make_apply_model_patches(
        family="public_api_test_d",
        family_apply=fam,
        module_path="transformers.models.llama.modeling_llama",
        classes={"decoder_layer": "LlamaDecoderLayer"},
        fused_add_rms_kind="public_api_fused",
    )
    assert callable(apply)


def test_raw_callable_accepted_without_registration():
    """Power-user shortcut: pass a factory callable directly, skip the registry."""
    fam = make_apply_family_patches(
        family="public_api_test_e",
        module_path="transformers.models.llama.modeling_llama",
    )
    apply = make_apply_model_patches(
        family="public_api_test_e",
        family_apply=fam,
        module_path="transformers.models.llama.modeling_llama",
        classes={"mlp": "LlamaMLP", "rms_norm": "LlamaRMSNorm"},
        activation_kind=_passthrough_factory,  # not registered, raw callable
        rms_norm_kind=_passthrough_factory,  # raw, also unregistered
    )
    assert callable(apply)


def test_activation_kwarg_gates_family_selected_activation_patch(monkeypatch):
    """The public fine-grained knob is ``activation``; the family-selected
    ``activation_kind`` decides which concrete kernel factory is installed."""
    module_name = "public_api_fake_activation_module"
    mod = types.ModuleType(module_name)

    class FakeMLP:
        def forward(self, x):
            return x

    mod.FakeMLP = FakeMLP
    monkeypatch.setitem(sys.modules, module_name, mod)

    calls = []

    def activation_factory(original):
        calls.append(original)

        def forward(self, x):
            return original(self, x)

        return forward

    fam = make_apply_family_patches(
        family="public_api_test_activation_gate",
        module_path=module_name,
    )
    apply = make_apply_model_patches(
        family="public_api_test_activation_gate",
        family_apply=fam,
        module_path=module_name,
        classes={"mlp": "FakeMLP"},
        activation_kind=activation_factory,
    )

    apply(performance=True, compat=True, activation=False)
    assert calls == []
    assert not hasattr(FakeMLP.forward, "__opaque_patched__")

    apply(performance=True, compat=True, activation=True)
    assert len(calls) == 1
    assert getattr(FakeMLP.forward, "__opaque_patched__", False)


def test_cross_entropy_sets_loss_function_on_model_instance(monkeypatch):
    import torch

    from opaque.api.patches.transformers.components.cross_entropy import (
        _opaque_causal_lm_loss,
    )

    module_name = "public_api_fake_loss_function_module"
    mod = types.ModuleType(module_name)

    def original_loss(*args, **kwargs):
        return None

    class FakeForCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_function = original_loss

        def forward(self, *args, **kwargs):
            return None

    mod.FakeForCausalLM = FakeForCausalLM
    monkeypatch.setitem(sys.modules, module_name, mod)

    fam = make_apply_family_patches(
        family="public_api_test_loss_function_patch",
        module_path=module_name,
    )
    apply = make_apply_model_patches(
        family="public_api_test_loss_function_patch",
        family_apply=fam,
        module_path=module_name,
        classes={"causal_lm": "FakeForCausalLM"},
    )

    patched = FakeForCausalLM()
    untouched = FakeForCausalLM()

    apply(patched, performance=True, compat=False, cross_entropy=False)
    assert patched.loss_function is original_loss

    apply(patched, performance=True, compat=False, cross_entropy=True)
    assert patched.loss_function is _opaque_causal_lm_loss
    assert untouched.loss_function is original_loss


def test_fused_linear_cross_entropy_gates_forward_independently(monkeypatch):
    """``fused_linear_cross_entropy=False`` keeps logits in the output while
    ``cross_entropy=True`` still installs the non-fused CE kernel."""
    import torch

    from opaque.api.patches.transformers.components.cross_entropy import (
        _opaque_causal_lm_loss,
    )

    module_name = "public_api_fake_fused_ce_module"
    mod = types.ModuleType(module_name)

    original_forward_sentinel = object()

    def original_loss(*args, **kwargs):
        return None

    class FakeForCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_function = original_loss

        def forward(self, *args, **kwargs):
            return original_forward_sentinel

    mod.FakeForCausalLM = FakeForCausalLM
    monkeypatch.setitem(sys.modules, module_name, mod)

    fam = make_apply_family_patches(
        family="public_api_test_fused_linear_ce",
        module_path=module_name,
    )
    apply = make_apply_model_patches(
        family="public_api_test_fused_linear_ce",
        family_apply=fam,
        module_path=module_name,
        classes={"causal_lm": "FakeForCausalLM"},
    )

    patched = FakeForCausalLM()

    # Opt out of only the fused linear+CE forward — loss_function should
    # still get the non-fused kernel and the class forward must NOT carry
    # the fused-CE marker (other concerns like kv_cache may still wrap
    # ``forward``; ``__opaque_patched__`` is the fused-CE-specific tag).
    apply(
        patched,
        performance=True,
        compat=False,
        fused_linear_cross_entropy=False,
    )
    assert patched.loss_function is _opaque_causal_lm_loss
    assert not hasattr(FakeForCausalLM.forward, "__opaque_patched__")

    # Historical ``cross_entropy=False`` cascades to disable the forward
    # patch as well — both stay off.
    fresh_mod = types.ModuleType(module_name + "_b")

    class FreshForCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_function = original_loss

        def forward(self, *args, **kwargs):
            return original_forward_sentinel

    fresh_mod.FreshForCausalLM = FreshForCausalLM
    monkeypatch.setitem(sys.modules, module_name + "_b", fresh_mod)

    fam_b = make_apply_family_patches(
        family="public_api_test_fused_linear_ce_b",
        module_path=module_name + "_b",
    )
    apply_b = make_apply_model_patches(
        family="public_api_test_fused_linear_ce_b",
        family_apply=fam_b,
        module_path=module_name + "_b",
        classes={"causal_lm": "FreshForCausalLM"},
    )

    fresh = FreshForCausalLM()
    apply_b(fresh, performance=True, compat=False, cross_entropy=False)
    assert fresh.loss_function is original_loss
    assert not hasattr(FreshForCausalLM.forward, "__opaque_patched__")


# ----------------------------------------------------------------------------
# family_name detection
# ----------------------------------------------------------------------------


def test_family_name_for_known_hf_model():
    import transformers

    cfg = transformers.AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
    cfg.num_hidden_layers = 1
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    assert family_name(model) == "qwen2"


def test_family_name_returns_none_for_non_hf_object():
    import torch

    assert family_name(torch.nn.Linear(2, 2)) is None


# ----------------------------------------------------------------------------
# make_apply_family_patches accepts overrides for the runtime replacements
# ----------------------------------------------------------------------------


def test_family_factory_default_replacements_are_opaque_vmap_safe():
    """Default closures bind opaque's vmap-safe implementations."""
    from opaque.api.patches.transformers._family import (
        apply_module_masking_patch,
        vmap_eager_attention_forward,
        vmap_repeat_kv,
        _opaque_apply_rotary_pos_emb,
    )

    apply = make_apply_family_patches(
        family="public_api_test_family_defaults",
        module_path="some.nonexistent.module",
    )
    # Sanity check that the closure references the expected defaults
    closure = {c.cell_contents for c in (apply.__closure__ or ())}
    assert vmap_repeat_kv in closure
    assert vmap_eager_attention_forward in closure
    assert _opaque_apply_rotary_pos_emb in closure
    assert apply_module_masking_patch in closure


def test_family_factory_accepts_custom_replacement_callables():
    """Power-user override: a custom architecture provides its own
    vmap-safe ``repeat_kv`` replacement."""
    sentinel_calls: list[tuple] = []

    def my_repeat_kv(*args, **kw):
        sentinel_calls.append((args, kw))
        return None

    apply = make_apply_family_patches(
        family="public_api_test_family_override",
        module_path="some.nonexistent.module",
        repeat_kv_replacement=my_repeat_kv,
    )
    assert callable(apply)
    closure = {c.cell_contents for c in (apply.__closure__ or ())}
    assert my_repeat_kv in closure


def test_family_factory_accepts_none_to_skip_concern():
    """Pass ``None`` per slot to skip that concern entirely (e.g. a
    non-RoPE architecture)."""
    apply = make_apply_family_patches(
        family="public_api_test_family_skip_rope",
        module_path="some.nonexistent.module",
        rope_replacement=None,
        masking_module_patcher=None,
    )
    closure = {c.cell_contents for c in (apply.__closure__ or ())}
    assert None in closure  # one of the skip-this slots


def test_family_factory_tracks_idempotency_per_enabled_concern(monkeypatch):
    from opaque.api.patches.transformers._family import _reset_patched_families

    _reset_patched_families()
    module_name = "public_api_fake_family_module"
    mod = types.ModuleType(module_name)

    def original_repeat_kv(x):
        return x

    def original_eager_attention_forward(x):
        return x

    def original_rope(x):
        return x

    def replacement_repeat_kv(x):
        return x

    def replacement_eager_attention_forward(x):
        return x

    def replacement_rope(x):
        return x

    mod.repeat_kv = original_repeat_kv
    mod.eager_attention_forward = original_eager_attention_forward
    mod.apply_rotary_pos_emb = original_rope
    monkeypatch.setitem(sys.modules, module_name, mod)

    apply = make_apply_family_patches(
        family="public_api_test_family_concern_tracking",
        module_path=module_name,
        repeat_kv_replacement=replacement_repeat_kv,
        eager_attention_replacement=replacement_eager_attention_forward,
        rope_replacement=replacement_rope,
        masking_module_patcher=None,
    )

    apply(performance=False, compat=False)
    assert mod.repeat_kv is original_repeat_kv
    assert mod.eager_attention_forward is original_eager_attention_forward
    assert mod.apply_rotary_pos_emb is original_rope

    apply(performance=False, compat=True)
    assert mod.repeat_kv is replacement_repeat_kv
    assert mod.eager_attention_forward is replacement_eager_attention_forward
    assert mod.apply_rotary_pos_emb is original_rope

    apply(performance=True, compat=False)
    assert mod.apply_rotary_pos_emb is replacement_rope


# ----------------------------------------------------------------------------
# register_family — user-defined families dispatch through the router
# ----------------------------------------------------------------------------


def test_register_family_makes_it_visible_in_supported_families(_restore_registry):
    name = "public_api_test_router_visibility"
    fam = make_apply_family_patches(family=name, module_path="some.fake.module")
    apply = make_apply_model_patches(
        family=name,
        family_apply=fam,
        module_path="some.fake.module",
        classes={},
        activation_kind="swiglu",
    )
    register_family(name, apply)
    assert name in supported_families()


def test_register_family_routes_via_apply_transformers_model_patches(_restore_registry):
    """``apply_transformers_model_patches`` (the dispatcher used by
    :func:`opaque.patches.apply_model_patches`) consults the registry,
    so user-registered families are reached without any source-code
    changes inside opaque-patches."""
    from opaque.api.patches.transformers._router import apply_transformers_model_patches

    name = "public_api_test_router_dispatch"
    calls: list[dict] = []

    def my_apply_fn(model, *, performance=True, compat=True, **kwargs):
        calls.append({"model": model, "performance": performance, "kwargs": kwargs})

    register_family(name, my_apply_fn)

    # Build a stub model that detect_family() will resolve to ``name``
    # via its config.model_type.
    class _Cfg:
        model_type = name

    class _Model:
        config = _Cfg()

    m = _Model()
    apply_transformers_model_patches(m, performance=False, compat=True, foo="bar")
    assert len(calls) == 1
    assert calls[0]["model"] is m
    assert calls[0]["performance"] is False
    assert calls[0]["kwargs"]["foo"] == "bar"


def test_unregistered_family_silently_skips():
    """An unknown family doesn't crash the dispatcher — it just doesn't
    apply any opaque patches."""
    from opaque.api.patches.transformers._router import apply_transformers_model_patches

    class _Cfg:
        model_type = "definitely-not-registered-xyz"

    class _Model:
        config = _Cfg()

    # Must not raise.
    apply_transformers_model_patches(_Model())
