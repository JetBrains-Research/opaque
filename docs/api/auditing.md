# Privacy Auditing

The `opaque.auditing` module provides functional tools for empirically auditing differential privacy guarantees using membership inference attacks on canary examples.

## Overview

Privacy auditing empirically validates DP guarantees by:

1. **Inserting canaries**: Training with (in) and without (out) specific examples
2. **Running attacks**: Computing membership scores for each canary
3. **Estimating epsilon**: Using statistical methods to bound the privacy parameter

**Key functions**:

- `epsilon_clopper_pearson()` - Conservative statistical bounds (recommended)
- `epsilon_one_run()` - Likelihood-ratio method from Nasr et al. (2023)
- `epsilon_raw_counts()` - Direct computation (less conservative)
- `audit()` - Convenience function for comprehensive auditing
- `attack_auroc()`, `tpr_at_fpr()`, `max_accuracy()` - Utility metrics

**See also**: [Privacy Auditing User Guide](../user-guide/auditing.md)

## Epsilon Estimation

### epsilon_clopper_pearson

```python
def epsilon_clopper_pearson(
    in_scores: Sequence[float] | np.ndarray,
    out_scores: Sequence[float] | np.ndarray,
    significance: float = 0.05,
    delta: float = 0.0,
    *,
    threshold: float | None = None,
) -> float:
```

Estimate epsilon using Clopper-Pearson confidence intervals.

Constructs conservative binomial confidence intervals for TPR/FPR and uses them to bound epsilon. Provides formal statistical guarantees with Bonferroni correction over all thresholds.

**Args**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_scores` | `Sequence[float] \| np.ndarray` | required | Attack scores for held-in canaries (training set) |
| `out_scores` | `Sequence[float] \| np.ndarray` | required | Attack scores for held-out canaries (test set) |
| `significance` | `float` | `0.05` | Allowed failure probability (1 - confidence) |
| `delta` | `float` | `0.0` | DP delta parameter (0 for pure DP) |
| `threshold` | `float \| None` | `None` | If provided, use this specific threshold instead of searching |

**Returns**: Epsilon lower bound at the specified confidence level.

**Example**:
```python
from opaque.auditing import epsilon_clopper_pearson

eps = epsilon_clopper_pearson(in_scores, out_scores, significance=0.05)
print(f"Epsilon lower bound (95% confidence): {eps:.2f}")
```

---

### epsilon_one_run

```python
def epsilon_one_run(
    in_scores: Sequence[float] | np.ndarray,
    out_scores: Sequence[float] | np.ndarray,
    significance: float = 0.05,
    delta: float = 0.0,
    *,
    threshold: float | None = None,
    eps_max: float = 20.0,
    tol: float = 1e-4,
) -> float:
```

Estimate epsilon using the one-run method from Nasr et al. (2023).

Uses a likelihood-ratio test tailored for DP auditing. Generally less conservative than Clopper-Pearson for the same sample size.

**Args**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_scores` | `Sequence[float] \| np.ndarray` | required | Attack scores for held-in canaries |
| `out_scores` | `Sequence[float] \| np.ndarray` | required | Attack scores for held-out canaries |
| `significance` | `float` | `0.05` | Allowed failure probability |
| `delta` | `float` | `0.0` | DP delta parameter |
| `threshold` | `float \| None` | `None` | If provided, use this specific threshold |
| `eps_max` | `float` | `20.0` | Maximum epsilon to search |
| `tol` | `float` | `1e-4` | Binary search tolerance |

**Returns**: Epsilon lower bound at the specified confidence level.

**Reference**: [Nasr et al. (2023)](https://arxiv.org/abs/2305.08846)

---

### epsilon_raw_counts

```python
def epsilon_raw_counts(
    in_scores: Sequence[float] | np.ndarray,
    out_scores: Sequence[float] | np.ndarray,
    min_count: int = 50,
    delta: float = 0.0,
) -> float:
```

Estimate epsilon from raw TPR/FPR counts.

Direct computation without confidence intervals. Less conservative but higher variance than Clopper-Pearson.

**Args**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_scores` | `Sequence[float] \| np.ndarray` | required | Attack scores for held-in canaries |
| `out_scores` | `Sequence[float] \| np.ndarray` | required | Attack scores for held-out canaries |
| `min_count` | `int` | `50` | Minimum FP count to consider a threshold |
| `delta` | `float` | `0.0` | DP delta parameter |

**Returns**: Epsilon estimate.

---

## Utility Metrics

### attack_auroc

```python
def attack_auroc(
    in_scores: Sequence[float] | np.ndarray,
    out_scores: Sequence[float] | np.ndarray,
) -> float:
```

Area under ROC curve for the membership inference attack.

- AUROC = 0.5 means random guessing (no privacy leakage detectable)
- AUROC = 1.0 means perfect attack (complete privacy breach)

**Example**:
```python
from opaque.auditing import attack_auroc

auroc = attack_auroc(in_scores, out_scores)
print(f"Attack AUROC: {auroc:.3f}")
```

---

### tpr_at_fpr

```python
def tpr_at_fpr(
    in_scores: Sequence[float] | np.ndarray,
    out_scores: Sequence[float] | np.ndarray,
    fpr: float | np.ndarray,
) -> float | np.ndarray:
```

True positive rate at a given false positive rate.

**Args**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `in_scores` | `Sequence[float] \| np.ndarray` | Attack scores for held-in canaries |
| `out_scores` | `Sequence[float] \| np.ndarray` | Attack scores for held-out canaries |
| `fpr` | `float \| np.ndarray` | Target false positive rate(s) in [0, 1] |

**Returns**: TPR value(s) at the specified FPR(s).

**Example**:
```python
from opaque.auditing import tpr_at_fpr

# Single FPR
tpr = tpr_at_fpr(in_scores, out_scores, fpr=0.01)
print(f"TPR at 1% FPR: {tpr:.3f}")

# Multiple FPRs
tprs = tpr_at_fpr(in_scores, out_scores, fpr=[0.001, 0.01, 0.1])
```

---

### max_accuracy

```python
def max_accuracy(
    in_scores: Sequence[float] | np.ndarray,
    out_scores: Sequence[float] | np.ndarray,
    *,
    prevalence: float | None = None,
) -> float:
```

Maximum classification accuracy achievable across all thresholds.

**Args**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_scores` | `Sequence[float] \| np.ndarray` | required | Attack scores for held-in canaries |
| `out_scores` | `Sequence[float] \| np.ndarray` | required | Attack scores for held-out canaries |
| `prevalence` | `float \| None` | `None` | Fraction of positives in population (default: use sample ratio) |

**Returns**: Maximum accuracy in [0, 1].

---

## Convenience Functions

### audit

```python
def audit(
    in_scores: Sequence[float] | np.ndarray,
    out_scores: Sequence[float] | np.ndarray,
    significance: float = 0.05,
    delta: float = 0.0,
    *,
    method: str = "clopper_pearson",
) -> AuditResult:
```

Run a comprehensive privacy audit.

Computes epsilon and all utility metrics in a single call.

**Args**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_scores` | array-like | required | Attack scores for held-in canaries |
| `out_scores` | array-like | required | Attack scores for held-out canaries |
| `significance` | `float` | `0.05` | Allowed failure probability |
| `delta` | `float` | `0.0` | DP delta parameter |
| `method` | `str` | `"clopper_pearson"` | Epsilon method: `"clopper_pearson"`, `"raw_counts"`, or `"one_run"` |

**Returns**: `AuditResult` namedtuple with fields:

| Field | Type | Description |
|-------|------|-------------|
| `epsilon` | `float` | Estimated epsilon lower bound |
| `auroc` | `float` | Attack AUROC |
| `tpr_at_low_fpr` | `float` | TPR at 1% FPR |
| `max_accuracy` | `float` | Maximum achievable accuracy |

**Example**:
```python
from opaque.auditing import audit

result = audit(in_scores, out_scores, significance=0.05, delta=1e-5)
print(f"Epsilon: {result.epsilon:.2f}")
print(f"AUROC: {result.auroc:.3f}")
print(f"TPR@1%FPR: {result.tpr_at_low_fpr:.3f}")
print(f"Max Accuracy: {result.max_accuracy:.3f}")

# Unpack as tuple
eps, auroc, tpr, acc = audit(in_scores, out_scores)
```

---

## Bootstrap Confidence Intervals

### bootstrap

```python
def bootstrap(
    fn: Callable,
    in_scores: Sequence[float] | np.ndarray,
    out_scores: Sequence[float] | np.ndarray,
    params: BootstrapParams,
) -> np.ndarray:
```

Compute bootstrapped quantiles for any auditing function.

Supports bias-corrected and accelerated (BCa) bootstrap for more accurate confidence intervals.

**Args**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `fn` | `Callable` | Function with signature `fn(in_scores, out_scores) -> float` |
| `in_scores` | array-like | Attack scores for held-in canaries |
| `out_scores` | array-like | Attack scores for held-out canaries |
| `params` | `BootstrapParams` | Bootstrap configuration |

**Returns**: Array of quantiles specified in `params.quantiles`.

**Example**:
```python
from opaque.auditing import bootstrap, attack_auroc, BootstrapParams

# Basic bootstrap
params = BootstrapParams(num_samples=1000, seed=42)
auroc_ci = bootstrap(attack_auroc, in_scores, out_scores, params)
print(f"AUROC 95% CI: [{auroc_ci[0]:.3f}, {auroc_ci[1]:.3f}]")

# With bias correction
params = BootstrapParams.confidence_interval(
    confidence=0.95,
    num_samples=2000,
    bias_correction=True,
    acceleration=True,
    seed=42,
)
eps_ci = bootstrap(
    lambda i, o: epsilon_clopper_pearson(i, o, significance=0.05),
    in_scores, out_scores, params
)
```

---

### BootstrapParams

```python
@dataclass(frozen=True)
class BootstrapParams:
    num_samples: int = 1000
    quantiles: tuple[float, ...] = (0.025, 0.975)
    seed: int | None = None
    bias_correction: bool = False
    acceleration: bool = False
```

Configuration for bootstrap confidence intervals.

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_samples` | `int` | `1000` | Number of bootstrap resamples |
| `quantiles` | `tuple[float, ...]` | `(0.025, 0.975)` | Quantiles to compute |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `bias_correction` | `bool` | `False` | Use bias-corrected bootstrap |
| `acceleration` | `bool` | `False` | Use accelerated (BCa) bootstrap |

**Factory method**:
```python
@classmethod
def confidence_interval(
    cls,
    confidence: float = 0.95,
    num_samples: int = 1000,
    bias_correction: bool = False,
    acceleration: bool = False,
    seed: int | None = None,
) -> BootstrapParams:
```

Create params for a symmetric confidence interval.

**Example**:
```python
# Default 95% CI
params = BootstrapParams()

# 99% CI with BCa
params = BootstrapParams.confidence_interval(
    confidence=0.99,
    num_samples=5000,
    bias_correction=True,
    acceleration=True,
)
```

---

## Quick Reference

| Function | Purpose | When to Use |
|----------|---------|-------------|
| `epsilon_clopper_pearson()` | Conservative epsilon bound | Default choice, formal guarantees |
| `epsilon_one_run()` | Tighter epsilon bound | Smaller sample sizes |
| `epsilon_raw_counts()` | Point estimate | Quick sanity checks |
| `attack_auroc()` | Attack strength | Compare attack methods |
| `tpr_at_fpr()` | Low-FPR performance | Real-world attack scenarios |
| `max_accuracy()` | Best-case attack | Upper bound on attack success |
| `audit()` | All metrics at once | Comprehensive reporting |
| `bootstrap()` | Confidence intervals | Uncertainty quantification |

---

## See Also

- **[Privacy Auditing User Guide](../user-guide/auditing.md)**: Conceptual explanations and workflows
- **[Tutorial: Empirical Privacy Auditing](../tutorials/07_privacy_auditing.ipynb)**: Interactive walkthrough
- **[Privacy Accounting](accounting.md)**: Theoretical privacy guarantees
