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
Holds the Pareto-optimal threshold structure and exposes query methods
for privacy metrics.

| Attribute | Type | Description |
|---|---|---|
| `n_in` | `int` | Number of held-in canaries |
| `n_out` | `int` | Number of held-out canaries |

### epsilon_at

```python
estimate.epsilon_at(*, delta=0.0, significance=0.05, threshold=None, eps_max=20.0, tol=1e-4) -> float
```

Epsilon lower bound using the one-run likelihood-ratio test (Steinke et al. 2023).
Tests positive-only, negative-only, and two-sided guesses per threshold, with
Bonferroni correction. When `threshold` is provided, uses that specific threshold
instead of searching over all Pareto-optimal thresholds.

### auc

```python
estimate.auc(*, confidence=None, num_samples=1000, key=None) -> float | tuple[float, float]
```

ROC AUC. Returns point estimate by default, or `(lower, upper)` CI tuple
when `confidence` is provided.

### beta_at

```python
estimate.beta_at(*, alpha) -> float | np.ndarray
```

Type-II error at given Type-I error rate. `beta = 1 - TPR` at `alpha = FPR`.

### summary

```python
estimate.summary(*, significance=0.05, delta=0.0, theoretical_epsilon=None) -> str
```

Formatted multi-line report. Includes `theoretical_epsilon` for comparison
when provided.

---

## Quick reference

| | |
|---|---|
| `auditing.coin_flip(dataset, ...)` | Coin-flip partition -> `CoinFlip` |
| `auditing.loss_scores(loss_fn, ...)` | Membership scores -> `np.ndarray` |
| `auditing.one_run(scores, coin_flip=cf)` | Estimate privacy -> `OneRunEstimate` |
| `cf.train_indices(len(dataset))` | Training indices for `dataset.select()` |
| `estimate.epsilon_at(delta=)` | Epsilon bound (one-run method) |
| `estimate.auc()` | Attack AUC |
| `estimate.auc(confidence=, key=)` | AUC with confidence interval |
| `estimate.beta_at(alpha=)` | Type-II error at given FPR |
| `estimate.summary(theoretical_epsilon=)` | Formatted report with comparison |
