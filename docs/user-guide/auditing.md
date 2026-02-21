# Privacy Auditing

This guide explains how to empirically validate differential privacy guarantees using membership inference attacks on canary examples.

## Overview

While privacy accounting provides theoretical guarantees, **privacy auditing** empirically validates them:

| Approach | What it tells you | Guarantees |
|----------|-------------------|------------|
| **Accounting** | Maximum possible privacy loss | Upper bound (theoretical) |
| **Auditing** | Actual observed privacy loss | Lower bound (empirical) |

A well-implemented DP system should have:
```
audited_epsilon <= theoretical_epsilon
```

If the audited epsilon exceeds the theoretical epsilon, there's likely a bug in your implementation.

## Quick Start

```python
import opaque.auditing as auditing
from opaque.random import key
from torch.utils.data import DataLoader

# 1. Setup: designate canaries and flip coins
experiment = auditing.setup(dataset, num_canaries=1000, key=key(42))

# 2. Train on the subset (excludes held-out canaries)
train_loader = DataLoader(experiment.subset(dataset), batch_size=32)
# ... standard DP-SGD training loop ...

# 3. Evaluate: score canaries and compute epsilon
audit = auditing.evaluate(experiment, loss_fn, params, dataset)
print(audit.summary())
```

Output:
```
Audit Summary
────────────────────────────────────────
  Samples:              502 in, 498 out
  AUROC:                0.7310
  ε (Clopper-Pearson):  1.2300
  ε (one-run):          1.6700
  TPR @ 1% FPR:         0.1200
  TPR @ 10% FPR:        0.3800
  Max accuracy:         0.6800
  (α=0.05, δ=0)
```

## How It Works (Steinke et al. 2023)

The **one-run** auditing method avoids training multiple models:

1. **Pick canaries**: Select m examples from the dataset as canaries
2. **Flip coins**: For each canary, flip a fair coin — include (heads) or exclude (tails)
3. **Train once**: Train on the full dataset minus excluded canaries
4. **Score**: Compute membership scores for all canaries (higher = more likely member)
5. **Test**: Count correct guesses and use a binomial test to bound epsilon

The key insight: non-canary data is always included. Only the m canaries are randomly in/out. This lets you audit with a single training run.

## End-to-End Example

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from opaque import clipped_grad, gaussian_noise, PoissonSampler
import opaque.auditing as auditing
from opaque.random import key

# Dataset
dataset = TensorDataset(X, y)

# ── Auditing setup (1 line) ──
experiment = auditing.setup(dataset, num_canaries=1000, key=key(42))

# ── Training (unchanged except: experiment.subset) ──
train_data = experiment.subset(dataset)
sampler = PoissonSampler(train_data, sample_rate=0.01)
train_loader = DataLoader(train_data, batch_sampler=sampler)

def loss_fn(params, x, y):
    return F.mse_loss(x @ params, y, reduction="sum")

grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=1)
noise_fn, noise_state = gaussian_noise(stddev=1.1 * grad_fn.clip_norm)

for step in range(num_steps):
    batch_x, batch_y = next(iter(train_loader))
    grads = grad_fn(params, batch_x, batch_y)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads

# ── Auditing evaluation (1 line) ──
audit = auditing.evaluate(experiment, loss_fn, params, dataset)

# ── Results ──
print(audit.summary(delta=1e-5))
print(f"Empirical epsilon: {audit.epsilon_at(delta=1e-5):.2f}")
```

## Epsilon Estimation Methods

### Clopper-Pearson (general)

Conservative statistical bounds using binomial confidence intervals. Works with **any** in/out split — no assumptions about how the split was created.

```python
audit.epsilon_at(delta=1e-5, method="clopper_pearson")
```

**When to use**: Post-hoc auditing with manual train/test splits, or when you want the most conservative bound.

### One-Run (Steinke/Nasr 2023)

Likelihood-ratio test tailored for coin-flip experiments. Tighter than Clopper-Pearson but **only valid when canaries were randomly included/excluded with probability 0.5**.

```python
audit.epsilon_at(delta=1e-5, method="one_run")
```

**When to use**: When using `auditing.setup()` + `auditing.evaluate()` (the standard path). This is the default.

### Automatic Selection

`epsilon_at()` automatically picks the right method based on how the `AuditResult` was created:

| Created via | Default method | Why |
|-------------|---------------|-----|
| `auditing.evaluate()` | `one_run` | Coin-flip setup is guaranteed |
| `AuditResult(in_scores, out_scores)` | `clopper_pearson` | No assumption about the split |

## Attack Metrics

Beyond epsilon, these metrics help understand attack strength:

```python
audit.auroc()                    # Area under ROC curve (0.5 = random, 1.0 = perfect)
audit.tpr_at_fpr(fpr=0.01)      # True positive rate at 1% false positive rate
audit.tpr_at_fpr(fpr=0.1)       # True positive rate at 10% FPR
audit.max_accuracy()             # Best-case classification accuracy
```

| Metric | Random | Weak Attack | Strong Attack |
|--------|--------|-------------|---------------|
| AUROC | 0.50 | 0.60 | 0.80+ |
| TPR @ 1% FPR | 0.01 | 0.05 | 0.20+ |
| Max accuracy | 0.50 | 0.60 | 0.80+ |

## Confidence Intervals with Bootstrap

```python
from opaque.auditing import AuditResult, BootstrapParams
from opaque.random import key

params = BootstrapParams.confidence_interval(
    confidence=0.95, num_samples=2000, key=key(42)
)

# Bootstrap any metric
auroc_ci = audit.bootstrap(AuditResult.auroc, params)
print(f"AUROC 95% CI: [{auroc_ci[0]:.3f}, {auroc_ci[1]:.3f}]")

# Bootstrap epsilon (use a lambda for parameterized metrics)
eps_ci = audit.bootstrap(
    lambda r: r.epsilon_at(delta=1e-5),
    params,
)
print(f"Epsilon 95% CI: [{eps_ci[0]:.2f}, {eps_ci[1]:.2f}]")
```

## Post-Hoc Auditing (without training integration)

If you already have membership scores from another source:

```python
from opaque.auditing import AuditResult

audit = AuditResult(in_scores, out_scores)
audit.epsilon_at(delta=1e-5)  # Uses Clopper-Pearson by default
print(audit.summary())
```

## Interpreting Results

### Healthy DP Implementation

```
Theoretical epsilon: 3.00
Audited epsilon:     1.50
```

The gap exists because: the attack may not be optimal, estimation is imperfect, and the bound is conservative.

### Potential Issues

```
Theoretical epsilon: 3.00
Audited epsilon:     5.00
```

Investigate: bug in gradient clipping, noise injection, privacy accounting, or data leakage.

## Best Practices

1. **Use enough canaries**: 1000+ for reliable results (100 is too few)
2. **Report confidence intervals**: Use `bootstrap()` to quantify uncertainty
3. **Audit multiple metrics**: Don't rely on epsilon alone
4. **Compare to theoretical epsilon**: The empirical bound should be lower

## API Reference

See [Privacy Auditing API Reference](../api/auditing.md) for detailed documentation.

## References

- Steinke, Nasr, Jagielski (2023). [Privacy Auditing with One (1) Training Run](https://arxiv.org/abs/2305.08846). NeurIPS 2023.
- Carlini et al. (2022). [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570).
