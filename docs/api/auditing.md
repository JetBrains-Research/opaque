# Privacy Auditing API

The `opaque.auditing` module provides empirical privacy auditing via membership inference on canary examples.

## Module-Level Functions

### auditing.setup

```python
def setup(
    dataset: Any,
    *,
    num_canaries: int,
    key: RngKey,
) -> CoinFlipExperiment:
```

Set up a one-run privacy audit experiment. Randomly selects canary examples and flips a fair coin for each to decide inclusion/exclusion (Steinke et al. 2023).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | any with `len()` | required | The full training dataset (HuggingFace or PyTorch) |
| `num_canaries` | `int` | required | Number of canary examples to designate |
| `key` | `RngKey` | required | RNG key for reproducibility |

**Returns**: `CoinFlipExperiment` managing the canary assignment.

```python
import opaque.auditing as auditing
from opaque.random import key

experiment = auditing.setup(dataset, num_canaries=1000, key=key(42))

# HuggingFace datasets:
train_data = dataset.select(experiment.train_indices(len(dataset)))

# PyTorch datasets:
train_data = experiment.subset(dataset)
```

---

### auditing.evaluate

```python
def evaluate(
    experiment: CoinFlipExperiment,
    loss_fn: Callable,
    *args: Any,
    batch_argnums: tuple[int, ...],
    dataset: Any,
    collate_fn: Callable | None = None,
    batch_unpack: Callable | None = None,
    batch_size: int = 256,
) -> AuditResult:
```

Score canaries by negative loss and produce audit results in one call. Uses `torch.func.vmap` for per-example loss computation. Follows the same `batch_argnums` convention as `clipped_grad`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment` | `CoinFlipExperiment` | required | From `auditing.setup()` |
| `loss_fn` | `Callable` | required | Per-example loss function, vmap-compatible |
| `*args` | any | | Non-batched args to `loss_fn` (e.g., model parameters) |
| `batch_argnums` | `tuple[int, ...]` | required | Indices of `loss_fn` args from dataset batches |
| `dataset` | any | required | The full dataset (same as passed to `setup`) |
| `collate_fn` | `Callable \| None` | `None` | Collate function for DataLoader |
| `batch_unpack` | `Callable \| None` | `None` | Maps batch to tuple of tensors for `batch_argnums` |
| `batch_size` | `int` | `256` | Batch size for scoring |

**Returns**: `AuditResult` with `epsilon_at()` defaulting to the `'one_run'` method.

```python
# HuggingFace pattern
audit = auditing.evaluate(
    experiment,
    per_example_loss_fn,
    trainable_params,
    batch_argnums=(1,),
    dataset=dataset,
    collate_fn=data_collator,
    batch_unpack=lambda b: (b["input_ids"].to(device),),
)

# PyTorch (x, y) pattern
audit = auditing.evaluate(
    experiment, loss_fn, params,
    batch_argnums=(1, 2),
    dataset=dataset,
)
```

---

### auditing.score

```python
def score(
    loss_fn: Callable,
    *args: Any,
    batch_argnums: tuple[int, ...],
    dataset: Any,
    indices: np.ndarray | None = None,
    collate_fn: Callable | None = None,
    batch_unpack: Callable | None = None,
    batch_size: int = 256,
) -> np.ndarray:
```

Compute membership scores as negative per-example loss. Higher score = lower loss = more likely a training member. Lower-level than `evaluate()` — use when you need raw scores.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_fn` | `Callable` | required | Per-example loss function, vmap-compatible |
| `*args` | any | | Non-batched args (e.g., model parameters) |
| `batch_argnums` | `tuple[int, ...]` | required | Which `loss_fn` args come from dataset batches |
| `dataset` | any | required | Dataset to score |
| `indices` | `np.ndarray \| None` | `None` | Score only these indices |
| `collate_fn` | `Callable \| None` | `None` | Collate function for DataLoader |
| `batch_unpack` | `Callable \| None` | `None` | Maps batch to tuple of tensors |
| `batch_size` | `int` | `256` | Batch size for scoring |

**Returns**: Array of scores, shape `(n,)`.

---

## AuditResult

```python
class AuditResult:
    n_in: int       # Number of held-in canaries
    n_out: int      # Number of held-out canaries
```

Privacy audit results computed from canary scores. Construct from `auditing.evaluate()` or directly from pre-computed scores.

### AuditResult.epsilon_at

```python
def epsilon_at(
    self,
    *,
    delta: float = 0.0,
    significance: float = 0.05,
    method: str | None = None,
) -> float:
```

Epsilon lower bound at the given delta. Matches the accounting API (`DpProcess.epsilon_at(delta=)`). Method is chosen automatically based on provenance.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delta` | `float` | `0.0` | DP delta parameter |
| `significance` | `float` | `0.05` | Failure probability (1 - confidence) |
| `method` | `str \| None` | auto | `'one_run'` or `'clopper_pearson'` |

### AuditResult.auc

```python
def auc(
    self,
    *,
    confidence: float | None = None,
    num_samples: int = 1000,
    key: RngKey | None = None,
) -> float | tuple[float, float]:
```

Area under the ROC curve. 0.5 = random guessing, 1.0 = perfect attack.

When `confidence` is provided, returns a confidence interval
as a `(lower, upper)` tuple instead of a point estimate.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence` | `float \| None` | `None` | If provided, return a symmetric CI at this level (e.g. 0.95 for 95% CI) |
| `num_samples` | `int` | `1000` | Number of resamples for CI |
| `key` | `RngKey \| None` | `None` | RNG key for reproducible resampling |

```python
audit.auc()                              # point estimate -> float
audit.auc(confidence=0.95, key=key(42))  # 95% CI -> (lower, upper)
```

### AuditResult.beta_at

```python
def beta_at(self, *, alpha: float | np.ndarray) -> float | np.ndarray:
```

Type-II error rate at a given Type-I error rate. Consistent with
`DpProcess.beta_at(alpha=)` in the accounting module. Higher beta means
the attack is weaker (more private). Relationship: `beta = 1 - TPR` at
`alpha = FPR`.

### AuditResult.max_accuracy

```python
def max_accuracy(self, *, prevalence: float | None = None) -> float:
```

Maximum classification accuracy across all thresholds.

### AuditResult.summary

```python
def summary(
    self,
    *,
    significance: float = 0.05,
    delta: float = 0.0,
    theoretical_epsilon: float | None = None,
) -> str:
```

Multi-line formatted summary of all metrics. When `theoretical_epsilon`
is provided, it is displayed alongside the empirical bound for comparison.

---

## CoinFlipExperiment

```python
class CoinFlipExperiment:
    num_canaries: int          # Total canary count
    canary_indices: np.ndarray # All canary dataset indices
    in_indices: np.ndarray     # Canaries included in training
    out_indices: np.ndarray    # Canaries excluded from training
```

Manages canary coin flips for one-run auditing. Prefer `auditing.setup()` over direct construction.

### CoinFlipExperiment.train_indices

```python
def train_indices(self, dataset_size: int) -> list[int]:
```

Return indices to use for training (all except held-out canaries). Returns
`list[int]` for direct use with HuggingFace `dataset.select()`.

### CoinFlipExperiment.subset

```python
def subset(self, dataset) -> torch.utils.data.Subset:
```

Return a `Subset` for training (excludes held-out canaries). For HuggingFace
datasets, prefer `train_indices()` with `dataset.select()` instead.

### CoinFlipExperiment.audit

```python
def audit(self, scores: np.ndarray) -> AuditResult:
```

Split scores by coin flip and return an `AuditResult`. The result defaults to the `'one_run'` epsilon method.

---

## Quick Reference

| Function / Method | Purpose |
|-------------------|---------|
| `auditing.setup()` | Set up canary experiment |
| `auditing.evaluate()` | Score canaries and compute audit |
| `auditing.score()` | Compute raw membership scores |
| `experiment.train_indices()` | Get training indices for HF `dataset.select()` |
| `experiment.subset()` | Get PyTorch `Subset` for training |
| `audit.epsilon_at(delta=)` | Epsilon bound (auto-selects method) |
| `audit.auc()` | Attack AUC (point estimate) |
| `audit.auc(confidence=0.95, key=)` | AUC with CI |
| `audit.beta_at(alpha=)` | Type-II error at given Type-I error |
| `audit.max_accuracy()` | Best-case attack accuracy |
| `audit.summary()` | Formatted report |
| `audit.summary(theoretical_epsilon=)` | Report with theoretical comparison |

## See Also

- **[Privacy Auditing User Guide](../user-guide/auditing.md)**: Conceptual explanations and workflows
- **[Privacy Auditing Tutorial](../tutorials/privacy_auditing.ipynb)**: Interactive walkthrough
