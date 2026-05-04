def apply_module_masking_patch(mod) -> bool:
    """Rebinds masking_utils functions in a model module's local namespace."""
    from opaque.patches.transformers.runtime.masking import (
        vmap_create_causal_mask,
        vmap_create_sliding_window_causal_mask,
    )

    patched = False
    if hasattr(mod, "create_causal_mask"):
        mod.create_causal_mask = vmap_create_causal_mask
        patched = True
    if hasattr(mod, "create_sliding_window_causal_mask"):
        mod.create_sliding_window_causal_mask = vmap_create_sliding_window_causal_mask
        patched = True
    return patched
