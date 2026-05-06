# Serialization

Opaque keeps training state in explicit values (clip state, noise state,
optimizer state, schedules, [`Accountant`](accounting.md) / [`DpProcess`](accounting.md),
…). Use :mod:`opaque.serialization` for a single **flat**
``dict[str, Any]`` suitable for :func:`torch.save` / :func:`torch.load`.

Restore is **template-driven**: pass a freshly constructed object of the same
shape as at save time; keys present in the checkpoint overwrite leaves, and
missing keys keep template values (forward compatibility when new fields appear).
Non-serialisable leaves (vendor specs, callables, …) are skipped on save and
preserved from the template on load.

Domain pages with examples: [Optimizers](optimizers.md), [Accounting](accounting.md).

The module also re-exports :func:`opaque.serialization.structural_state_dict` and
:func:`opaque.serialization.structural_from_state_dict` (same tree logic without the
type registry).

::: opaque.serialization
    options:
      show_source: true
      heading_level: 2
