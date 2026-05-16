# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
def _disable_kv_cache(forward_fn):
    """Wrap forward to disable KV cache when no existing cache is passed.

    HuggingFace models default to ``config.use_cache = True``, which
    allocates a ``DynamicCache`` on every forward pass — wasted memory
    during training, where the cache is built but never consumed.  Under
    ``vmap`` the same cache objects also leak via circular references
    (cache → tensors → autograd metadata → cache) that defeat refcounting,
    so the patch doubles as a vmap-safety win even though it sits in the
    performance bucket.

    This wrapper forces ``use_cache=False`` when ``past_key_values`` is
    ``None`` or an empty cache, matching the approach used by Unsloth
    (``if past_key_values is None and self.training: use_cache = False``).
    Unlike Unsloth, we don't gate on ``self.training`` to avoid side effects
    with LoRA modules — the condition ``past_key_values is None`` is
    sufficient since the cache is only useful for autoregressive generation
    with an existing cache.
    """
    import functools

    if getattr(forward_fn, "_opaque_cache_disabled", False):
        return forward_fn

    @functools.wraps(forward_fn)
    def wrapper(*args, **kwargs):
        past = kwargs.get("past_key_values")
        has_cached_data = past is not None and (
            not hasattr(past, "get_seq_length") or past.get_seq_length() > 0
        )
        if not has_cached_data:
            kwargs["use_cache"] = False
        return forward_fn(*args, **kwargs)

    wrapper._opaque_cache_disabled = True
    return wrapper


def apply_kv_cache_patch(target_cls: type | None, model=None) -> None:
    """Apply the KV cache disabler wrapper to a specific model class and instance."""
    if target_cls is None:
        return

    import types

    # Global class-level patch
    if hasattr(target_cls, "forward"):
        fwd = target_cls.forward
        if not getattr(fwd, "_opaque_cache_disabled", False):
            new_fwd = _disable_kv_cache(fwd)
            target_cls.forward = new_fwd

    # Instance-level patch
    if model is not None:
        for module in model.modules():
            if type(module) is target_cls:
                fwd = module.forward
                unbound = getattr(fwd, "__func__", fwd)
                if not getattr(unbound, "_opaque_cache_disabled", False):
                    new_unbound = _disable_kv_cache(unbound)
                    module.forward = types.MethodType(new_unbound, module)
