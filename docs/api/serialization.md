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

:class:`~opaque.types.PerGroup` checkpoints through the same API (flat keys such
as ``groups.<param_key>`` / ``values.<group_name>``).  NumPy ``ndarray`` leaves
are supported alongside ``torch.Tensor``.

Domain pages with examples: [Optimizers](optimizers.md), [Accounting](accounting.md).

::: opaque.serialization
    options:
      show_source: true
      heading_level: 2
