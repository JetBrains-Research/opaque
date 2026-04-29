# Learning-rate Scheduling

Standard ML LR schedules — linear / cosine decay with warmup — work
without modification under DP-SGD: they scale the already-clipped,
already-noised update direction.  The privacy guarantee is
unchanged.

This page explains the small composition primitive Opaque adds on top
of [`torchopt.schedule`](https://torchopt.readthedocs.io/) and the
pattern for using it.

## Why warmup matters under DP-SGD

The first few hundred steps of DP-SGD update parameters using a
gradient direction that's heavily contaminated by Gaussian noise — at
typical noise multipliers, the signal-to-noise ratio per step is
small even when the average over many steps is informative.  A high
LR early in training amplifies that noise into the parameters.  A
short linear ramp from `0` to `base_lr` over the first few percent of
total steps gives the optimizer's moment estimators time to average
the noise out before they're scaled by the full step size.  The same
schedule that's a nice-to-have for non-private fine-tuning becomes
load-bearing under DP.

Empirically, anywhere from **3% to 10% of total steps** as warmup is
typical for fine-tuning under DP — closer to 10% when the noise
multiplier is large.

## Constant LR

```python
import torchopt

opt = torchopt.adamw(lr=1e-3)
```

A float passes straight through TorchOpt; no schedule object is
needed.

## Pure decay

```python
import torchopt

# Linear decay 1e-3 -> 0 over 10 000 steps.
schedule = torchopt.schedule.linear_schedule(1e-3, 0.0, transition_steps=10_000)
opt = torchopt.adamw(lr=schedule)
```

TorchOpt's `linear_schedule`, `polynomial_schedule`, and
`exponential_decay` cover the common decay curves; reach for them
first.  Opaque adds [`cosine_schedule`](../api/schedules.md#cosine_schedule)
and [`inverse_sqrt_schedule`](../api/schedules.md#inverse_sqrt_schedule)
for the cases TorchOpt doesn't ship.

## Adding warmup

[`with_warmup`](../api/schedules.md#with_warmup) multiplies a
schedule by a linear ramp `0 → 1` during the first `transition_steps`
steps and leaves it untouched afterwards.

For the standard "warmup, then decay" shape, configure the decay
with `transition_begin = num_warmup_steps`.  Schedules in
`opaque.dpsgd.schedules` and `torchopt.schedule` return their
`init_value` while `step < transition_begin`, so the multiplicative
ramp turns that leading plateau into the warmup ramp.

```python
import torchopt
from opaque.dpsgd.schedules import with_warmup

base_lr, W, N = 1e-3, 500, 10_000

decay = torchopt.schedule.linear_schedule(
    base_lr, 0.0,
    transition_steps=N - W,
    transition_begin=W,
)
schedule = with_warmup(decay, transition_steps=W)
opt = torchopt.adamw(lr=schedule)
```

`with_warmup` raises `ValueError` for `transition_steps <= 0`, so
gate it on a runtime warmup length:

```python
schedule = (
    with_warmup(decay, transition_steps=W) if W > 0 else decay
)
```

The default warmup ramp is linear; `ramp="cosine"`, `ramp="1-sqrt"`,
or any `Callable[[float], float]` mapping `progress ∈ [0, 1]` to a
factor in `[0, 1]` selects a different shape:

```python
schedule = with_warmup(decay, transition_steps=W, ramp="cosine")
```

## Three worked examples

### Linear with warmup

```python
import torchopt
from opaque.dpsgd.schedules import with_warmup

base_lr, W, N = 1e-3, 500, 10_000
decay = torchopt.schedule.linear_schedule(
    base_lr, 0.0, transition_steps=N - W, transition_begin=W,
)
schedule = with_warmup(decay, transition_steps=W)
```

### Cosine with warmup

```python
from opaque.dpsgd.schedules import with_warmup, cosine_schedule

base_lr, W, N = 1e-3, 500, 10_000
decay = cosine_schedule(
    base_lr, 0.0, transition_steps=N - W, transition_begin=W,
)
schedule = with_warmup(decay, transition_steps=W)
```

### Constant after warmup

`with_warmup` accepts a scalar `float` as shorthand for a constant
schedule, so warm-up-then-plateau is a one-liner:

```python
from opaque.dpsgd.schedules import with_warmup

schedule = with_warmup(1e-3, transition_steps=500)
```

## Restarts

[`with_restarts`](../api/schedules.md#with_restarts) replays a
schedule `num_cycles` times over a window.  Combined with cosine you
get SGDR (Loshchilov & Hutter, 2017):

```python
from opaque.dpsgd.schedules import cosine_schedule, with_restarts

base_lr, N, k = 1e-3, 4000, 4
cycle_length = N / k

inner = cosine_schedule(base_lr, 0.0, transition_steps=cycle_length)
schedule = with_restarts(inner, transition_steps=N, num_cycles=k)
```

Each cycle the schedule snaps back to `inner(0)` and runs the curve
over its own `cycle_length`.  Composes with `with_warmup` the same
way as any other decay.

## Reading the current LR

Schedules are plain callables — `schedule(step)` returns the LR at
any step:

```python
for step in range(num_steps):
    # ... clipping, noise, update ...
    if step % 100 == 0:
        print(f"step={step}  lr={schedule(step):.6f}")
```

The optimizer maintains its own counter inside `opt_state` (via
`scale_by_schedule`), so calling `schedule(step)` for logging does
not interfere with how the optimizer applies the schedule.

## See Also

- [Schedules API](../api/schedules.md) — full signatures and behavior.
- [Optimizers User Guide](optimizers.md) — bias correction, weight decay, AdamW-BC.
