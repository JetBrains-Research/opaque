# Schedules

Pure-Python step-indexed scalar schedules.  Each public function
returns a plain `Callable[[int], float]` that plugs straight into
`torchopt.adamw(lr=...)`, `torchopt.adam(lr=...)`,
`torchopt.sgd(lr=...)`, or [`opaque.optimizers.adamw`](optimizers.md).
TorchOpt's `scale_by_neg_lr` advances the schedule via the
optimizer-state step counter — no manual `.step()` call.

Public surface:

- pure curves: `constant_schedule`, `linear_schedule`,
  `polynomial_schedule`, `exponential_decay`, `cosine_schedule`,
  `inverse_sqrt_schedule`, `one_minus_sqrt_schedule`;
- composition primitives: `with_warmup`, `with_restarts`.

---

## Composing a schedule

A decay curve is configured by `transition_begin` (when decay starts)
and `transition_steps` (how long it takes); it returns its
`init_value` during `[0, transition_begin)` and decays during
`[transition_begin, transition_begin + transition_steps]`.  Wrap the
decay with `with_warmup` to multiply it by a `0 → 1` ramp during the
first `transition_steps` steps — turning the leading plateau into a
warmup ramp.

```python
import torchopt
from opaque.scheduling import with_warmup, cosine_schedule

W, N, base_lr = 100, 10000, 1e-3

# Cosine that plateaus during [0, W), then decays over [W, N).
decay = cosine_schedule(
    init_value=base_lr, end_value=0.0,
    transition_steps=N - W, transition_begin=W,
)

# Replace the plateau with a 0 -> base_lr ramp.
schedule = with_warmup(decay, transition_steps=W)

opt = torchopt.adamw(lr=schedule)
```

---

## `constant_schedule`

```python
constant_schedule(value: float) -> Callable[[int], float]
```

Returns `value` at every step.  Equivalent to passing a float
directly to TorchOpt's `lr` argument; `with_warmup` accepts a float
as the same shorthand.

```python
from opaque.scheduling import constant_schedule, with_warmup

schedule = constant_schedule(1e-3)

# Warmup, then plateau.
schedule = with_warmup(1e-3, transition_steps=100)
```

---

## `linear_schedule`

```python
linear_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> Callable[[int], float]
```

Linear interpolation from `init_value` to `end_value` over
`transition_steps`, starting at `transition_begin`.  Steps before
`transition_begin` hold at `init_value`; steps after the transition
hold at `end_value`.

```python
from opaque.scheduling import linear_schedule

# 1e-3 -> 0 over 1000 steps.
sched = linear_schedule(1e-3, 0.0, transition_steps=1000)
```

---

## `polynomial_schedule`

```python
polynomial_schedule(
    init_value: float,
    end_value: float,
    power: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> Callable[[int], float]
```

Polynomial transition from `init_value` to `end_value`:
`end + (init - end) * (1 - count/T)^power`.  `power=1` reduces to
[`linear_schedule`](#linear_schedule); larger `power` produces a
flatter early phase and steeper late drop.

```python
from opaque.scheduling import polynomial_schedule

sched = polynomial_schedule(1e-3, 1e-7, power=2.0, transition_steps=1000)
```

---

## `exponential_decay`

```python
exponential_decay(
    init_value: float,
    decay_rate: float,
    transition_begin: int = 0,
    transition_steps: int = 1,
    staircase: bool = False,
    end_value: float | None = None,
) -> Callable[[int], float]
```

Geometric decay: `init * decay_rate^((step - transition_begin) / transition_steps)`.
With `staircase=True`, the exponent is floored, so the schedule
drops in discrete jumps every `transition_steps`.  `end_value`, when
set, clamps the result (max for `decay_rate < 1`, min otherwise).

```python
from opaque.scheduling import exponential_decay

# Halve LR every 1000 steps, but never below 1e-5.
sched = exponential_decay(1e-3, decay_rate=0.5, transition_steps=1000, end_value=1e-5)
```

---

## `cosine_schedule`

```python
cosine_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
    num_cycles: float = 0.5,
) -> Callable[[int], float]
```

Cosine annealing from `init_value` to `end_value` over
`transition_steps`.  Steps before `transition_begin` hold at
`init_value`.  `num_cycles=0.5` (the default) is a single half-cosine
that bottoms out at `end_value` exactly when `progress == 1`; values
greater than `0.5` produce additional oscillations clamped at zero.

```python
from opaque.scheduling import cosine_schedule

# Single half-cosine 1e-3 -> 0 over 1000 steps.
sched = cosine_schedule(1e-3, 0.0, transition_steps=1000)

# 1.5 oscillations over 1000 steps.
sched = cosine_schedule(1e-3, 0.0, transition_steps=1000, num_cycles=1.5)
```

---

## `inverse_sqrt_schedule`

```python
inverse_sqrt_schedule(
    init_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> Callable[[int], float]
```

Inverse-square-root decay
`init_value * sqrt(T / (s + T))`, where `T = transition_steps` is the
timescale and `s = max(0, step - transition_begin)`.  At `s = 0`
returns `init_value`; at `s = T` returns `init_value / sqrt(2)`; at
`s = 3T` returns `init_value / 2`.

```python
from opaque.scheduling import inverse_sqrt_schedule

# Decay from 1e-3 with timescale 1000.  At step 1000: ~7.07e-4.
sched = inverse_sqrt_schedule(1e-3, transition_steps=1000)
```

---

## `one_minus_sqrt_schedule`

```python
one_minus_sqrt_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
    transition_begin: int = 0,
) -> Callable[[int], float]
```

Decay following `factor = 1 - sqrt(progress)` from `init_value` at
`transition_begin` to `end_value` at
`transition_begin + transition_steps`.  Concave decreasing — drops
faster early than late.  Held at `end_value` after the transition.

```python
from opaque.scheduling import one_minus_sqrt_schedule

# Drops 1e-3 -> 1e-5 with concave shape over 1000 steps.
sched = one_minus_sqrt_schedule(1e-3, 1e-5, transition_steps=1000)
```

---

## `with_warmup`

```python
with_warmup(
    schedule: Callable[[int], float] | float,
    transition_steps: int,
    *,
    ramp: str | Callable[[float], float] = "linear",
) -> Callable[[int], float]
```

Multiply `schedule` by a 0 → 1 ramp over the first `transition_steps`
steps; afterwards return `schedule(step)` unchanged.  A scalar
`float` for `schedule` is treated as `constant_schedule(value)`.

The `ramp` kwarg picks the warmup curve shape:

- `"linear"` (default): `progress`
- `"cosine"`: `0.5 * (1 - cos(pi * progress))`
- `"1-sqrt"`: `1 - sqrt(1 - progress)`
- a callable `f(progress)` taking progress in `[0, 1]` and returning
  a factor in `[0, 1]` — for any custom shape.

All schedules in this module return `init_value` for
`step < transition_begin`, so configuring a decay with
`transition_begin == transition_steps` of the warmup produces the
standard "warmup, then decay" shape:

| step range                                         | result                              |
|----------------------------------------------------|-------------------------------------|
| `[0, transition_steps)`                            | `(step / transition_steps) * init_value` (linear ramp 0 → init_value) |
| `[transition_steps, decay's transition_begin)`     | `init_value` (plateau, only if there's a gap) |
| later                                              | `schedule(step)` as configured      |

```python
from opaque.scheduling import with_warmup, linear_schedule, cosine_schedule

base_lr, W, N = 1e-3, 100, 1000
decay_steps = N - W

# Linear-decay with warmup.
sched = with_warmup(
    linear_schedule(base_lr, 0.0, decay_steps, transition_begin=W),
    transition_steps=W,
)

# Cosine-decay with warmup.
sched = with_warmup(
    cosine_schedule(base_lr, 0.0, decay_steps, transition_begin=W),
    transition_steps=W,
)

# Warmup-then-constant via the float shorthand.
sched = with_warmup(base_lr, transition_steps=W)
```

Raises `ValueError` if `transition_steps <= 0` or `ramp` is an
unknown string.

---

## `with_restarts`

```python
with_restarts(
    schedule: Callable[[int], float],
    transition_steps: int,
    num_cycles: int,
    transition_begin: int = 0,
) -> Callable[[int], float]
```

Repeat `schedule` `num_cycles` times across
`[transition_begin, transition_begin + transition_steps)`.  Each
cycle has length `transition_steps / num_cycles`; within a cycle,
`schedule` is evaluated at the cycle-local step
(`relative_step % cycle_length`).  Configure `schedule` to produce
its full curve over a single cycle of that length.

Before `transition_begin` returns `schedule(0)`; after the final
cycle, returns `schedule(cycle_length)`.

```python
from opaque.scheduling import cosine_schedule, with_restarts

# SGDR: cosine annealing repeated 4 times over 4000 steps.
inner = cosine_schedule(1e-3, 0.0, transition_steps=1000)
sched = with_restarts(inner, transition_steps=4000, num_cycles=4)
```

Raises `ValueError` if `num_cycles <= 0` or `transition_steps <= 0`.

---

## See Also

- [Optimizers API](optimizers.md) — pass any of these schedules to TorchOpt's `lr` argument.
- [LR Scheduling User Guide](../user-guide/lr-scheduling.md) — patterns and DP-specific guidance.
