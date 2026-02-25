# Auditing API

::: opaque.auditing

## Module functions

### setup

```python
auditing.setup(
    dataset, *, num_canaries, key,
    batch_argnums=None, collate_fn=None,
    batch_unpack=None, batch_size=256,
) -> CoinFlipExperiment
```

Randomly select canaries, flip coins, and optionally store scoring config
so that `evaluate()` requires only the loss function and trained parameters.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset` | any with `len()` | required | Full training dataset |
| `num_canaries` | `int` | required | Number of canaries to designate |
| `key` | `RngKey` | required | RNG key for reproducibility |
| `batch_argnums` | `tuple[int, ...]` | `None` | Which `loss_fn` args are batched (same as `clipped_grad`) |
| `collate_fn` | `Callable` | `None` | DataLoader collate function |
| `batch_unpack` | `Callable` | `None` | Extract tensors from collated batch |
| `batch_size` | `int` | `256` | Scoring batch size |

When `batch_argnums` and other scoring params are provided, the experiment
stores them so `evaluate()` can use them automatically.

```python
# Recommended: configure scoring at setup time
experiment = auditing.setup(
    dataset, num_canaries=1000, key=key(42),
    batch_argnums=(1,),
    collate_fn=data_collator,
    batch_unpack=lambda b: (b["input_ids"].to(device),),
)
```

---

### evaluate

```python
auditing.evaluate(
    experiment, loss_fn, *args, *,
    batch_argnums=..., dataset=...,
    collate_fn=..., batch_unpack=...,
    batch_size=None,
) -> AuditResult
```

Score all canaries and return an `AuditResult`. Parameters fall back to
values stored at `setup()` time when not provided explicitly.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `experiment` | `CoinFlipExperiment` | required | From `setup()` |
| `loss_fn` | `Callable` | required | Per-example loss (vmap-compatible) |
| `*args` | | | Non-batched args (e.g. model params) |
| `batch_argnums` | `tuple[int, ...]` | from setup | Which `loss_fn` args are batched |
| `dataset` | any | from setup | Full dataset |
| `collate_fn` | `Callable` | from setup | DataLoader collate function |
| `batch_unpack` | `Callable` | from setup | Extract tensors from collated batch |
| `batch_size` | `int` | from setup | Scoring batch size |

```python
# If setup() has scoring config, evaluate is a one-liner:
audit = auditing.evaluate(experiment, loss_fn, trained_params)

# Or override specific params:
audit = auditing.evaluate(experiment, loss_fn, params, batch_size=32)
```

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
than `evaluate()` — use when you need raw scores.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `loss_fn` | `Callable` | required | Per-example loss (vmap-compatible) |
| `*args` | | | Non-batched args (e.g. model params) |
| `batch_argnums` | `tuple[int, ...]` | required | Which `loss_fn` args are batched |
| `dataset` | any | required | Dataset to score |
| `indices` | `np.ndarray` | `None` | Score only these indices |
| `collate_fn` | `Callable` | `None` | DataLoader collate function |
| `batch_unpack` | `Callable` | `None` | Extract tensors from collated batch |
| `batch_size` | `int` | `256` | Scoring batch size |

**Returns** `np.ndarray` of shape `(n,)`. Higher = more likely member.

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
audit.epsilon_at(*, delta=0.0, significance=0.05, method=None) -> float
```

Epsilon lower bound. Auto-selects method: `'one_run'` if created via
`CoinFlipExperiment.audit()`, `'clopper_pearson'` otherwise.

### epsilon_clopper_pearson

```python
audit.epsilon_clopper_pearson(*, significance=0.05, delta=0.0, threshold=None) -> float
```

Conservative binomial CI bound. Bonferroni-corrected over Pareto-optimal
thresholds unless `threshold` is given.

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

## CoinFlipExperiment

```python
class CoinFlipExperiment(canary_indices, *, key)
```

Prefer `auditing.setup()` over direct construction.

| Attribute | Type | Description |
|---|---|---|
| `num_canaries` | `int` | Total canary count |
| `canary_indices` | `np.ndarray` | All canary dataset indices |
| `in_indices` | `np.ndarray` | Included in training (heads) |
| `out_indices` | `np.ndarray` | Excluded from training (tails) |

### train_indices

```python
experiment.train_indices(dataset_size) -> list[int]
```

All indices except held-out canaries. Returns `list[int]` for HuggingFace
`dataset.select()`.

### subset

```python
experiment.subset(dataset) -> torch.utils.data.Subset
```

PyTorch `Subset` for training. For HuggingFace datasets, prefer
`train_indices()` with `dataset.select()`.

### audit

```python
experiment.audit(scores) -> AuditResult
```

Split scores by coin flip. `scores` must have shape `(num_canaries,)`,
one per canary in the order of `canary_indices`.

---

## Quick reference

| | |
|---|---|
| `auditing.setup(..., batch_argnums=, ...)` | Designate canaries + configure scoring |
| `auditing.evaluate(exp, loss_fn, params)` | Score canaries, return `AuditResult` |
| `auditing.score()` | Raw membership scores |
| `experiment.train_indices()` | Training indices for `dataset.select()` |
| `experiment.subset()` | PyTorch `Subset` for training |
| `audit.epsilon_at(delta=)` | Epsilon bound (auto method) |
| `audit.auc()` | Attack AUC |
| `audit.auc(confidence=, key=)` | AUC with confidence interval |
| `audit.beta_at(alpha=)` | Type-II error at given FPR |
| `audit.summary(theoretical_epsilon=)` | Formatted report with comparison |
