# Serialization

Opaque keeps training state in explicit values (clip state, noise state,
optimizer state, schedules, [`Accountant`](accounting.md) / [`DpProcess`](accounting.md),
…). Use `opaque.serialization` for a single **flat**
`dict[str, Any]` suitable for `torch.save` / `torch.load`.

Restore is **template-driven**: pass a freshly constructed object of the same
shape as at save time; keys present in the checkpoint overwrite leaves, and
missing keys keep template values (forward compatibility when new fields appear).

Dispatch resolves a leaf by exact type and then by `__mro__`, so a subclass
of a registered type — a custom `torch.Tensor` subclass, for example — is
serialized by the nearest base-class handler rather than dropped.
(`torch.nn.Parameter` is common enough to get its own exact-type handler that
preserves the subclass and its `requires_grad` flag on restore, so it does
not rely on the `__mro__` fallback.) A leaf that is neither registered nor a
generic container (dataclass / NamedTuple / tuple / list / mapping) nor a
primitive raises `TypeError` on both save and restore instead of being
silently skipped.
Genuinely inert leaves that the template reproduces (vendor structure handles
such as `optree.PyTreeSpec`) are declared with
`opaque.serialization.register_template_restored`; nothing is written for
them and the template supplies them on load.

`PerGroup` checkpoints through the same API. Path keys serialize as their
`str()` form under the structural walker — for example, `groups.('a',)` for a
flat leaf, or `groups.('layer', 'weight')` for a nested path — alongside
`values.<group_name>`. When DP bias correction is enabled on Adam-family
optimizers, `phi` is a path-keyed dict from `opt.init` so `from_state_dict`
round-trips without resetting φ. NumPy `ndarray` leaves are supported alongside
`torch.Tensor`.

Domain pages with examples: [Optimizers](optimizers.md), [Accounting](accounting.md).

## API surface (no per-type `state_dict()`)

Opaque centralizes (de)serialization in `opaque.serialization.state_dict`
and `opaque.serialization.from_state_dict`. Registered types (including
`Accountant`) are written and restored **only**
through these module-level functions — there are no instance methods named
`state_dict` / `from_state_dict` on those classes.

Minimal round trip for an `Accountant`:

```python
from opaque.accounting import Accountant, identity
from opaque.serialization import from_state_dict, state_dict

acct = Accountant() | identity()
flat = state_dict(acct)
acct2 = from_state_dict(Accountant(), flat)
```

The same pattern applies to clip state, noise state, functional optimizer
state, and any other value that flows through the DP training loop.  Custom
types may register handlers with `opaque.serialization.register_serializer`,
or — when a leaf carries no run state and the template reproduces it —
declare it inert with `opaque.serialization.register_template_restored`.

::: opaque.serialization
    options:
      show_source: true
      heading_level: 2
