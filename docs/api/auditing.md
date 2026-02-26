# Auditing API

::: opaque.auditing

## Module functions

### setup

```python
auditing.setup(
    dataset, *,
    num_canaries=None, key=None,
    coin_flip=None,
    batch_argnums=None, collate_fn=None,
    batch_unpack=None, batch_size=256,
) -> OneRunEstimator
```

Set up a one-run privacy audit. Creates (or accepts) a coin-flip
partition and wraps it with the dataset and scoring configuration.

Either provide `num_canaries` + `key` to create a partition
automatically, or provide a pre-built `coin_flip`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset` | any with `len()` | required | Full training dataset |
| `num_canaries` | `int` | `None` | Number of canaries to designate |
| `key` | `RngKey` | `None` | RNG key for reproducibility |
| `coin_flip` | `CoinFlip` | `None` | Pre-built partition (overrides num_canaries/key) |
| `batch_argnums` | `tuple[int, ...]` | `None` | Which `loss_fn` args are batched (same as `clipped_grad`) |
| `collate_fn` | `Callable` | `None` | DataLoader collate function |
| `batch_unpack` | `Callable` | `None` | Extract tensors from collated batch |
| `batch_size` | `int` | `256` | Scoring batch size |

```python
# Create partition automatically
audit_state = auditing.setup(
    dataset, num_canaries=1000, key=key(42),
    batch_argnums=(1,),
)
train_data = dataset.select(audit_state.train_indices)

# Or with a pre-built CoinFlip
cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
audit_state = auditing.setup(dataset, coin_flip=cf, batch_argnums=(1,))
```

---

### coin_flip

```python
auditing.coin_flip(
    dataset, *, num_canaries, key,
) -> CoinFlip
```

Create a coin-flip partition. Randomly selects `num_canaries` examples
and flips a fair coin for each. Use when you want to inspect or reuse
the partition before calling `setup()`.

---

### score

```python
auditing.score(
    loss_fn, *args, *,
    batch_argnums, dataset,
    indices=None, collate_fn=None,
    batch_unpack=None, batch_size=256,
) -> np.ndarray
```

Compute per-example membership scores as negative loss. Lower-level
utility — most users should use `OneRunEstimator.evaluate()` instead.

**Returns** `np.ndarray` of shape `(n,)`. Higher = more likely member.

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

## OneRunEstimator

```python
class OneRunEstimator
```

One-run estimator returned by `auditing.setup()`. Call `evaluate()`
after training to score canaries and produce an `AuditResult`.

| Attribute | Type | Description |
|---|---|---|
| `coin_flip` | `CoinFlip` | The coin-flip partition |
| `train_indices` | `list[int]` | Dataset indices for training (excludes held-out canaries) |

### evaluate

```python
estimator.evaluate(loss_fn, *args, *, batch_argnums=None, batch_size=None) -> AuditResult
```

Score canaries and produce audit results. Uses the scoring config
stored at `setup()` time. Optional `batch_argnums` and `batch_size`
override stored values.

```python
audit_state = auditing.setup(dataset, num_canaries=1000, key=key(42),
                             batch_argnums=(1,), ...)
train_data = dataset.select(audit_state.train_indices)
# ... train ...
result = audit_state.evaluate(loss_fn, trained_params)
```

### audit

```python
estimator.audit(scores) -> AuditResult
```

Split pre-computed scores by coin flip and return an `AuditResult`.
Use `evaluate()` for the full pipeline (scoring + splitting).

---

## AuditResult

```python
class AuditResult(in_scores, out_scores)
```

| Attribute | Type | Description |
|---|---|---|
| `n_in` | `int` | Number of held-in canaries |
| `n_out` | `int` | Number of held-out canaries |

### epsilon_at

```python
audit.epsilon_at(*, delta=0.0, significance=0.05) -> float
```

Epsilon lower bound. Uses the one-run likelihood-ratio test.

### epsilon_one_run

```python
audit.epsilon_one_run(*, significance=0.05, delta=0.0, threshold=None, eps_max=20.0, tol=1e-4) -> float
```

Likelihood-ratio test from Steinke et al. (2023). Tests both positive-only
and two-sided guesses per threshold, with Bonferroni correction.

### auc

```python
audit.auc(*, confidence=None, num_samples=1000, key=None) -> float | tuple[float, float]
```

ROC AUC. Returns point estimate by default, or `(lower, upper)` CI tuple
when `confidence` is provided.

### beta_at

```python
audit.beta_at(*, alpha) -> float | np.ndarray
```

Type-II error at given Type-I error rate. `beta = 1 - TPR` at `alpha = FPR`.

### max_accuracy

```python
audit.max_accuracy(*, prevalence=None) -> float
```

Best-case classification accuracy across all thresholds.

### summary

```python
audit.summary(*, significance=0.05, delta=0.0, theoretical_epsilon=None) -> str
```

Formatted multi-line report. Includes `theoretical_epsilon` for comparison
when provided.

---

## Quick reference

| | |
|---|---|
| `auditing.setup(dataset, ...)` | Designate canaries + configure scoring -> `OneRunEstimator` |
| `auditing.coin_flip(dataset, ...)` | Coin-flip partition only -> `CoinFlip` |
| `auditing.score(...)` | Raw membership scores |
| `audit_state.evaluate(loss_fn, params)` | Score canaries, return `AuditResult` |
| `audit_state.train_indices` | Training indices for `dataset.select()` |
| `result.epsilon_at(delta=)` | Epsilon bound (one-run method) |
| `result.auc()` | Attack AUC |
| `result.auc(confidence=, key=)` | AUC with confidence interval |
| `result.beta_at(alpha=)` | Type-II error at given FPR |
| `result.summary(theoretical_epsilon=)` | Formatted report with comparison |
