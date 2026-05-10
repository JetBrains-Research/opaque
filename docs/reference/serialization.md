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

## API surface (no per-type `state_dict()`)

Opaque centralises (de)serialisation in :func:`opaque.serialization.state_dict`
and :func:`opaque.serialization.from_state_dict`.  Registered types (including
:class:`~opaque.accounting.Accountant`) are written and restored **only**
through these module-level functions — there are no instance methods named
``state_dict`` / ``from_state_dict`` on those classes.

Minimal round-trip for an :class:`~opaque.accounting.Accountant`:

```python
from opaque.accounting import Accountant, identity
from opaque.serialization import from_state_dict, state_dict

acct = Accountant() | identity()
flat = state_dict(acct)
acct2 = from_state_dict(Accountant(), flat)
```

The same pattern applies to clip state, noise state, functional optimizer
state, and any other value that flows through the DP training loop.  Custom
types may register handlers with :func:`opaque.serialization.register_serializer`.

::: opaque.serialization
    options:
      show_source: true
      heading_level: 2
