# opaque-auditing

Empirical privacy auditing for Opaque: one-run estimator
(Steinke et al. 2023), coin-flip canary injection, and
loss-based membership-inference attacks.

## Install

Install the root package as described in the
[repository installation guide](../../README.md#installation).
Use its `auditing` extra to include this component.

`opaque-auditing` depends on `opaque-engine` and `scipy`; both install
automatically with the extra.

## Quick start

```python
import opaque.auditing as auditing
from opaque.random import key

cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
# ... DP-SGD training loop ...
estimate = auditing.one_run(scores, coin_flip=cf)
```

## Layout

- `opaque.auditing.one_run` — one-run estimator (Steinke et al. 2023)
- `opaque.auditing.coin_flip()` — coin-flip canary injection
- `opaque.auditing.loss_scores()` — loss-based membership inference via `vmap`
