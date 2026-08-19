# opaque.torch

The PyTorch provider's own public surface — helpers that are inherently
torch-shaped and therefore live in the `opaque-torch` wheel rather than the
neutral engine. `opaque.torch.functional.make_functional` and
`opaque.torch.random` are documented with
[Utilities](utilities.md) and [RNG](rng.md); this page covers the
distributed, device, and checkpoint-compat surfaces.

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

::: opaque.torch.device.DeviceCapabilities
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
`opaque.patches.apply_runtime_patches` composes these; they are exposed here
so integrations can install a single patch without the whole patch set.

::: opaque.torch.checkpoint
    options:
        show_source: false
        heading_level: 3
        members: true
        filters: ["!^_"]
