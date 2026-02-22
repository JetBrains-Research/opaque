# Privacy Auditing

Privacy accounting provides theoretical upper bounds on privacy loss.
Privacy auditing empirically validates those bounds by running a
membership inference attack. If the audited epsilon exceeds the theoretical
epsilon, there is likely a bug in the implementation.

## How it works

Opaque implements the one-run auditing method from
[Steinke, Nasr, Jagielski (2023)](https://arxiv.org/abs/2305.08846):

1. **Designate canaries.** Select `m` examples from the dataset as
   canaries.
2. **Flip coins.** For each canary, independently include (heads) or
   exclude (tails) it from the training set with probability 0.5.
3. **Train once.** Train on the resulting subset. Non-canary data is always
   included.
4. **Score.** Compute a membership score for each canary (higher = more
   likely a member). The default score is the loss difference between a
   canary and a reference.
5. **Test.** Use a binomial test to bound epsilon from the scores and the
   coin flips.

The key advantage is that only one training run is needed, unlike methods
that require training hundreds of shadow models.

## Quick start

```python
import opaque.auditing as auditing
from opaque.random import key
from torch.utils.data import DataLoader

# 1. Setup: designate canaries and flip coins
experiment = auditing.setup(dataset, num_canaries=1000, key=key(42))

# 2. Train on the subset (excludes held-out canaries)
train_data = experiment.subset(dataset)
train_loader = DataLoader(train_data, batch_size=32)
# ... standard DP-SGD training loop on train_data ...

# 3. Evaluate: score canaries and compute epsilon
audit = auditing.evaluate(experiment, loss_fn, params, dataset)
print(audit.summary())
```

## End-to-end example

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import torchopt
from opaque import clipped_grad, gaussian_noise, PoissonSampler
import opaque.auditing as auditing
from opaque.random import key

dataset = TensorDataset(X, y)

# Auditing setup
experiment = auditing.setup(dataset, num_canaries=1000, key=key(42))

# Training (only change: use experiment.subset)
train_data = experiment.subset(dataset)
sampler = PoissonSampler(train_data, sample_rate=0.01, key=key(0))
train_loader = DataLoader(train_data, batch_sampler=sampler)

def loss_fn(params, x, y):
    return F.mse_loss(x @ params, y, reduction="sum")

grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=(1, 2))
noise_fn, noise_state = gaussian_noise(
    stddev=1.1 * clip_state.sensitivity(), key=key(42),
)

optimizer = torchopt.sgd(lr=0.01)
opt_state = optimizer.init(params)

for step in range(num_steps):
    batch_x, batch_y = next(iter(train_loader))
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer.update(noisy_grads, opt_state)
    params = torchopt.apply_updates(params, updates)

# Evaluate
audit = auditing.evaluate(experiment, loss_fn, params, dataset)
print(f"Empirical epsilon: {audit.epsilon_at(delta=1e-5):.2f}")
```

## Epsilon estimation methods

### One-run (default)

A likelihood-ratio test tailored for the coin-flip setup. Tighter than
Clopper-Pearson but only valid when canaries were randomly included or
excluded with probability 0.5 (the standard `auditing.setup` path).

```python
audit.epsilon_at(delta=1e-5, method="one_run")
```

### Clopper-Pearson

Conservative binomial confidence intervals. Works with any in/out split,
including manual splits not created by `auditing.setup`.

```python
audit.epsilon_at(delta=1e-5, method="clopper_pearson")
```

`epsilon_at()` automatically selects the appropriate method based on how
the `AuditResult` was created:

| Created via | Default method |
|-------------|---------------|
| `auditing.evaluate()` | `one_run` |
| `AuditResult(in_scores, out_scores)` | `clopper_pearson` |

## Attack metrics

Beyond epsilon, these metrics quantify the strength of the membership
inference attack:

```python
audit.auc()                    # Area under ROC curve (0.5 = random)
audit.beta_at(alpha=0.01)       # Type-II error at 1% Type-I error
audit.beta_at(alpha=0.1)        # Type-II error at 10% Type-I error
audit.max_accuracy()             # Best-case classification accuracy
```

## Confidence intervals

Quantify uncertainty in the AUC estimate:

```python
from opaque.random import key

auc_ci = audit.auc(confidence=0.95, key=key(42))
print(f"AUC 95% CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
```

## Post-hoc auditing

If you already have membership scores from another source (e.g., a
different attack or a different framework):

```python
from opaque.auditing import AuditResult

audit = AuditResult(in_scores, out_scores)
audit.epsilon_at(delta=1e-5)  # uses Clopper-Pearson by default
print(audit.summary())
```

## Interpreting results

A healthy DP implementation shows an audited epsilon below the theoretical
epsilon:

```
Theoretical epsilon: 3.00
Audited epsilon:     1.50   # expected: gap exists because the attack is not optimal
```

If the audited epsilon exceeds the theoretical epsilon, investigate:
a bug in gradient clipping, noise injection, privacy accounting, or data
leakage outside the training loop.

## Practical guidance

- **Use enough canaries.** 1000+ gives reliable results. 100 is too few
  for tight bounds.
- **Report confidence intervals.** Use `auc(confidence=0.95)` to quantify
  uncertainty.
- **Compare to theoretical epsilon.** The empirical bound should be lower.
- **Audit multiple metrics.** AUC and beta at low alpha are complementary
  to epsilon.

## References

- Steinke, Nasr, Jagielski (2023). [Privacy Auditing with One (1) Training Run](https://arxiv.org/abs/2305.08846). NeurIPS 2023.
- Carlini et al. (2022). [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570).

## API reference

See [Auditing API Reference](../api/auditing.md) for complete function
signatures and return types.
