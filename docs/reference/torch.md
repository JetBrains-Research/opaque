# opaque.torch

The PyTorch provider's own public surface — helpers that are inherently
torch-shaped and therefore live in the `opaque-torch` wheel rather than the
neutral engine. `opaque.torch.functional.make_functional` and
`opaque.torch.random` are documented with
[Utilities](utilities.md) and [RNG](rng.md); this page covers the
distributed, device, checkpoint-compat, and functional-transform
introspection surfaces.

## In-place distributed collectives

Return-based collectives live on the neutral
[`opaque.distributed`](distributed.md) façade. The torch provider adds the
in-place variants DDP training loops use to avoid re-allocating gradients.

::: opaque.torch.distributed.all_reduce_
    options:
        show_source: true
        heading_level: 3

::: opaque.torch.distributed.reduce_pytree_
    options:
        show_source: true
        heading_level: 3

::: opaque.torch.distributed.sum_gradients_
    options:
        show_source: true
        heading_level: 3

## Device capabilities

::: opaque.torch.device.device_capabilities
    options:
        show_source: true
        heading_level: 3

::: opaque.torch.device.types.DeviceCapabilities
    options:
        heading_level: 3

::: opaque.torch.device.fused_kernels_available
    options:
        show_source: true
        heading_level: 3

::: opaque.torch.device.sdpa_autocast_under_vmap_broken
    options:
        show_source: true
        heading_level: 3

## Checkpoint compatibility

Vmap-safe gradient-checkpointing patch installers and probes.
`apply_checkpoint_patch()` installs the set, and is what
[`opaque.execution`](execution.md)'s `checkpoint` and
`optimize_saved_activations` need before they compose under a functional
transform. The individual installers are exposed for integrations that want one
patch rather than the whole set. `opaque.torch.apply_runtime_patches` selects the
whole set as its one concern; higher layers forward to it before applying
runtime patches of their own, so a caller makes a single call.

::: opaque.torch.checkpoint
    options:
        show_source: false
        heading_level: 3
        members: true
        filters: ["!^_"]

## Functional-transform introspection

A patch that must behave differently inside a `torch.func` transform asks
here rather than probing `torch._C._functorch` itself.

::: opaque.torch.transforms.under_functorch_transform
    options:
        show_source: true
        heading_level: 3
