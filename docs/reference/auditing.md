# Auditing API

::: opaque.auditing

## Module functions

### coin_flip

```python
auditing.coin_flip(
    dataset, *, num_canaries, key,
) -> CoinFlip
```

Create a coin-flip partition. Randomly selects `num_canaries` examples
and flips a fair coin for each to decide inclusion/exclusion.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset` | any with `len()` | required | Full training dataset |
| `num_canaries` | `int` | required | Number of canaries to designate |
| `key` | `RngKey` | required | RNG key for reproducibility |

```python
cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
train_data = dataset.select(cf.train_indices(len(dataset)))
```

---

### loss_scores

```python
auditing.loss_scores(
    loss_fn, *args, *,
    batch_argnums, dataloader,
    reference_scores=None,
) -> np.ndarray
```

Compute per-example membership scores as negative loss. Higher score =
lower loss = more likely a training member.

The `dataloader` must yield batches compatible with `loss_fn`. Each batch
should be a tensor (single `batch_argnums`) or a tuple of tensors
(multiple `batch_argnums`). Use a custom `collate_fn` on the DataLoader
to handle dict-style batches (e.g., HuggingFace).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loss_fn` | `Callable` | required | Per-example loss function (vmap-compatible) |
| `*args` | any | — | Non-batched arguments (e.g., model parameters) |
| `batch_argnums` | `tuple[int, ...]` | required | Which `loss_fn` args are batched (same as `clipped_grad`) |
| `dataloader` | iterable | required | Yields tensors or tuples of tensors |
| `reference_scores` | `np.ndarray` | `None` | Baseline scores to subtract (e.g., from untrained model) |

**Returns** `np.ndarray` of shape `(n,)`. Higher = more likely member.

```python
from torch.utils.data import DataLoader, Subset

def canary_collate(examples):
    batch = data_collator(examples)
    return (batch["input_ids"].to(device),)

canary_loader = DataLoader(
    Subset(dataset, cf.canary_indices.tolist()),
    batch_size=32, collate_fn=canary_collate,
)
scores = auditing.loss_scores(
    loss_fn, params,
    batch_argnums=(1,),
    dataloader=canary_loader,
)
```

---

### gradient_scores

```python
auditing.gradient_scores(
    loss_fn, *args, *,
    batch_argnums, dataloader,
    reference_scores=None,
) -> np.ndarray
```

Compute per-example membership scores as negative squared gradient norm.
Higher score = smaller gradient norm = more likely a training member.

This is a white-box attack that differentiates with respect to the first
`loss_fn` argument (model parameters). Therefore `0` must not appear in
`batch_argnums`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loss_fn` | `Callable` | required | Per-example scalar loss function |
| `*args` | any | — | Non-batched arguments; first arg is differentiated |
| `batch_argnums` | `tuple[int, ...]` | required | Batched argument indices; must exclude 0 |
| `dataloader` | iterable | required | Yields tensors or tuples of tensors |
| `reference_scores` | `np.ndarray` | `None` | Baseline scores to subtract |

**Returns** `np.ndarray` of shape `(n,)`. Higher = more likely member.

```python
ref = auditing.gradient_scores(
    loss_fn, initial_params,
    batch_argnums=(1,),
    dataloader=canary_loader,
)
scores = auditing.gradient_scores(
    loss_fn, trained_params,
    batch_argnums=(1,),
    dataloader=canary_loader,
    reference_scores=ref,
)
```

The scorer is also available from the attacks namespace:

```python
from opaque.auditing.attacks import gradient_scores
```

---

### one_run

```python
auditing.one_run(scores, *, coin_flip) -> OneRunEstimate
```

Build a one-run privacy estimate from canary scores. Splits scores by
the coin-flip partition, precomputes the Pareto-optimal ROC frontier,
and returns a frozen estimate.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `scores` | `np.ndarray` | required | Per-canary membership scores, shape `(num_canaries,)` |
| `coin_flip` | `CoinFlip` | required | The coin-flip partition |

```python
estimate = auditing.one_run(scores, coin_flip=cf)
print(estimate.epsilon_at(delta=1e-5))
```

---

## CoinFlip

```python
class CoinFlip(canary_indices, *, key)
```

Coin-flip partitioning for canary-based privacy auditing. Each canary
is independently included or excluded with probability 0.5.

| Attribute | Type | Description |
|---|---|---|
| `num_canaries` | `int` | Total canary count |
| `canary_indices` | `np.ndarray` | All canary dataset indices |
| `in_indices` | `np.ndarray` | Included in training (heads) |
| `out_indices` | `np.ndarray` | Excluded from training (tails) |

### train_indices

```python
cf.train_indices(dataset_size) -> list[int]
```

All indices except held-out canaries. Returns `list[int]` for HuggingFace
`dataset.select()`.

### split_scores

```python
cf.split_scores(scores) -> tuple[np.ndarray, np.ndarray]
```

Split per-canary scores into `(in_scores, out_scores)`.

---

## OneRunEstimate

```python
class OneRunEstimate  # frozen dataclass
```

Precomputed one-run audit estimate, returned by `auditing.one_run()`.
Holds the Pareto-optimal threshold structure and exposes:

- a **default audit-method surface** (`epsilon_at`, `delta_at`, `beta_at`,
  `advantage`) that dispatches to μ-GDP — the paper-recommended default
  for Gaussian-DP mechanisms;
- two **method factories** (`eps_delta`, `gdp`) for explicit override;
- **attack-side empirical metrics** (`attack_auc`, `attack_beta_at`).

| Attribute | Type | Description |
|---|---|---|
| `n_in` | `int` | Number of held-in canaries |
| `n_out` | `int` | Number of held-out canaries |

### epsilon_at

```python
estimate.epsilon_at(*, delta, significance=0.05, threshold=None) -> float
```

Epsilon lower bound from the default μ-GDP audit method. Equivalent to
`self.gdp().epsilon_at(...)`. Requires `delta > 0`. For non-Gaussian-DP
mechanisms use `self.eps_delta().epsilon_at(...)` explicitly.

### delta_at

```python
estimate.delta_at(*, epsilon, significance=0.05, threshold=None) -> float
```

δ(ε) from the default μ-GDP audit method. Equivalent to
`self.gdp().delta_at(...)`.

### beta_at

```python
estimate.beta_at(*, alpha, significance=0.05, threshold=None) -> float
```

Theoretical f-DP β at α under the inferred μ̂-GDP. Equivalent to
`self.gdp().beta_at(...)`. For the empirical attack ROC β, see
[`attack_beta_at`](#attack_beta_at).

### advantage

```python
estimate.advantage(*, significance=0.05, threshold=None) -> float
```

Total-variation advantage at the inferred μ̂-GDP. Equivalent to
`self.gdp().advantage()`.

### eps_delta

```python
estimate.eps_delta() -> EpsDeltaMethod
```

Mechanism-agnostic (ε, δ)-DP order-statistics audit method
(Xiang et al. 2025). Use for non-Gaussian-DP mechanisms or pure ε-DP
auditing. See [EpsDeltaMethod](#epsdeltamethod) below.

### gdp

```python
estimate.gdp(*, grid_size=10_000) -> GdpMethod
```

μ-GDP order-statistics audit method (Xiang et al. 2025) with a tunable
integration grid. The top-level `epsilon_at` / `delta_at` / `beta_at` /
`advantage` are shortcuts for `gdp()` with the default `grid_size`. See
[GdpMethod](#gdpmethod) below.

### attack_auc

```python
estimate.attack_auc(*, confidence=None, num_samples=1000, key=None) -> float | tuple[float, float]
```

Empirical ROC AUC of the membership inference attack. Returns point
estimate by default, or `(lower, upper)` CI tuple when `confidence` is
provided. Independent of the audit method.

### attack_beta_at

```python
estimate.attack_beta_at(*, alpha) -> float | np.ndarray
```

Empirical attack β at given FPR: `1 − TPR` interpolated from the
Pareto-optimal ROC frontier. Distinct from
[`beta_at`](#beta_at) which is the theoretical f-DP β at the
inferred μ̂-GDP.

---

## EpsDeltaMethod

Returned by `estimate.eps_delta()`. Exposes the two query directions
along the audit's (ε, δ) boundary.

### epsilon_at

```python
method.epsilon_at(*, delta=0.0, significance=0.05, threshold=None) -> float
```

Largest ε the audit can certify at the given δ. `delta=0` is allowed
(pure ε-DP). When `threshold` is provided, accuracy is evaluated at that
specific score threshold; otherwise the Pareto-optimal threshold
maximising TP + TN is used.

### delta_at

```python
method.delta_at(*, epsilon, significance=0.05, threshold=None) -> float
```

Largest δ at which the audit certifies ε ≥ `epsilon`. Returns `0.0` when
`epsilon` is unreachable even at δ=0.

`EpsDeltaMethod` deliberately does not expose `beta_at` / `advantage`:
the (ε, δ)-DP trade-off function is a family envelope, so those metrics
would be worst-case across the family rather than instance-specific —
use `GdpMethod` for sharp f-DP queries on Gaussian-DP mechanisms.

---

## GdpMethod

Returned by `estimate.gdp(grid_size=)`. Mirrors the full
[`Pld`](accounting.md) metric surface: all four methods derive from a
single inferred μ̂.

| Attribute | Type | Default | Description |
|---|---|---|---|
| `grid_size` | `int` | `10_000` | Grid points for the order-statistics integration |

### epsilon_at

```python
method.epsilon_at(*, delta, significance=0.05, threshold=None) -> float
```

Largest ε the audit can certify at the given δ. Requires `delta > 0`
(μ-GDP is incompatible with pure ε-DP).

### delta_at

```python
method.delta_at(*, epsilon, significance=0.05, threshold=None) -> float
```

δ(ε) under the inferred μ̂-GDP guarantee, via the closed-form GDP relation.

### beta_at

```python
method.beta_at(*, alpha, significance=0.05, threshold=None) -> float
```

f-DP Type-II error at α under the inferred μ̂-GDP:
β(α; μ) = Φ(Φ⁻¹(1 − α) − μ̂). Theoretical β of the post-audit guarantee,
distinct from `OneRunEstimate.attack_beta_at` (empirical attack ROC).

### advantage

```python
method.advantage(*, significance=0.05, threshold=None) -> float
```

Total-variation advantage at the inferred μ̂-GDP: TV(μ) = 2·Φ(μ̂/2) − 1.

---

## Quick reference

| | |
|---|---|
| `auditing.coin_flip(dataset, ...)` | Coin-flip partition → `CoinFlip` |
| `auditing.loss_scores(loss_fn, ...)` | Membership scores → `np.ndarray` |
| `auditing.gradient_scores(loss_fn, ...)` | White-box membership scores → `np.ndarray` |
| `auditing.one_run(scores, coin_flip=cf)` | Estimate privacy → `OneRunEstimate` |
| `cf.train_indices(len(dataset))` | Training indices for `dataset.select()` |
| `cf.canary_subset(dataset)` | `Subset` of canary examples for DataLoader |
| `estimate.epsilon_at(delta=)` | ε bound (μ-GDP default) |
| `estimate.delta_at(epsilon=)` | δ at given ε |
| `estimate.beta_at(alpha=)` | Theoretical β at α (μ-GDP) |
| `estimate.advantage()` | TV advantage at inferred μ̂ |
| `estimate.eps_delta()` | Mechanism-agnostic (ε, δ)-DP method (override) |
| `estimate.gdp(grid_size=)` | μ-GDP method with tunable grid |
| `estimate.attack_auc()` | Empirical attack AUC |
| `estimate.attack_auc(confidence=, key=)` | Empirical AUC with CI |
| `estimate.attack_beta_at(alpha=)` | Empirical attack β at given FPR |
