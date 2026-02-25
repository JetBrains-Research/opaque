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
   likely a member). The default score is negative loss.
5. **Test.** Use a likelihood-ratio test to bound epsilon from the scores
   and the coin flips.

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
#    HuggingFace datasets:
train_data = dataset.select(experiment.train_indices(len(dataset)))
#    Or PyTorch datasets:
train_data = experiment.subset(dataset)
# ... standard DP-SGD training loop on train_data ...

# 3. Evaluate: score canaries and compute epsilon
audit = auditing.evaluate(
    experiment,
    per_example_loss_fn,
    trainable_params,
    batch_argnums=(1,),
    dataset=dataset,
    collate_fn=data_collator,
    batch_unpack=lambda b: (b["input_ids"].to(device),),
)
print(audit.summary(delta=1e-5))
```

## End-to-end example with HuggingFace

```python
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

import opaque.auditing as auditing
from opaque import clipped_grad, gaussian_noise, PoissonSampler, make_functional
from opaque.random import key, fold_in

# ... load and tokenize HF dataset into train_dataset ...
# ... load model, apply LoRA, convert to functional form ...

# Auditing setup
experiment = auditing.setup(train_dataset, num_canaries=1000, key=key(42))
full_dataset = train_dataset  # Keep reference to full dataset
train_dataset = train_dataset.select(experiment.train_indices(len(train_dataset)))

# Recalculate sample rate with reduced dataset
sample_rate = batch_size / len(train_dataset)

# Standard DP-SGD training
sampler = PoissonSampler(train_dataset, sample_rate=sample_rate, key=key(0))
train_loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=data_collator)

def per_example_loss_fn(params, tokens):
    output = fmodel(merged_params(params), tokens, labels=tokens)
    return output.loss

grad_fn, clip_state = clipped_grad(
    per_example_loss_fn, argnums=0, batch_argnums=(1,),
    l2_clip_norm=1.0, keep_batch_dim=True,
)
noise_fn, noise_state = gaussian_noise(
    stddev=1.1 * clip_state.sensitivity(), key=key(42),
)

for batch in train_loader:
    tokens = batch["input_ids"].to(device)
    (grads, aux), clip_state = grad_fn(params, tokens, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer.update(noisy_grads, opt_state)
    params = torchopt.apply_updates(params, updates)

# Evaluate
audit = auditing.evaluate(
    experiment,
    per_example_loss_fn,
    params,
    batch_argnums=(1,),
    dataset=full_dataset,
    collate_fn=data_collator,
    batch_unpack=lambda b: (b["input_ids"].to(device),),
)
print(f"Empirical epsilon: {audit.epsilon_at(delta=1e-5):.2f}")
print(audit.summary(delta=1e-5, theoretical_epsilon=target_epsilon))
```

## End-to-end example with PyTorch

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

# Training (use experiment.subset for PyTorch datasets)
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
audit = auditing.evaluate(
    experiment, loss_fn, params,
    batch_argnums=(1, 2),
    dataset=dataset,
)
print(f"Empirical epsilon: {audit.epsilon_at(delta=1e-5):.2f}")
```

## Scoring functions

The default scoring function computes the per-example training loss: canaries
that the model has memorized will have lower loss than canaries that were
excluded. This loss-based score works well for most settings.

The `batch_argnums` parameter follows the same convention as
`clipped_grad`, specifying which arguments of the loss function come
from dataset batches and should be vmapped over.

For HuggingFace models that produce dict-style batches, use `batch_unpack`
to extract the relevant tensors:

```python
audit = auditing.evaluate(
    experiment,
    per_example_loss_fn,
    params,
    batch_argnums=(1,),
    dataset=dataset,
    collate_fn=data_collator,
    batch_unpack=lambda b: (b["input_ids"].to(device),),
)
```

**Choosing a score:**

| Score | When to use |
|-------|-------------|
| Training loss (default) | General-purpose, works well for most models |
| Loss decrease from init | Stronger signal (paper's recommended black-box score) |
| Custom scoring | When loss is not well-defined |

Loss-based scores tend to give the tightest epsilon bounds because they
directly measure memorization signal.

## Canary selection and statistical power

### Number of canaries

The number of canaries directly affects the statistical power of the audit.
More canaries means tighter confidence intervals on the estimated epsilon.

| Canaries | Typical precision |
|----------|-------------------|
| 100 | Very noisy, wide confidence intervals |
| 500 | Usable for detecting large bugs |
| 1000+ | Recommended for reliable epsilon bounds |
| 5000+ | Tight bounds, needed for small-epsilon regimes |

Rule of thumb: use at least 500 canaries. For $\varepsilon < 1$, use 2000+.

### Random vs targeted canaries

`auditing.setup` selects canaries uniformly at random. This gives an unbiased
estimate of average-case privacy. For worst-case auditing, you can manually
select outlier examples (rare classes, unusual features) as canaries and
construct the experiment manually.

## Epsilon estimation methods

### One-run (default)

A likelihood-ratio test from Steinke et al. (2023), Corollary 5.4.
For each Pareto-optimal threshold, tests both positive-only guesses
and two-sided guesses (counting both correct inclusions and correct
exclusions), taking the best result.  Only valid when canaries were
randomly included or excluded with probability 0.5 (the standard
`auditing.setup` path).

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

### Understanding the gap

The gap between theoretical and empirical epsilon is expected and normal.
Theoretical accounting provides an *upper bound*; the audit runs an actual
attack, which is suboptimal. Common reasons for a large gap:

- **Weak attack:** Loss-based membership inference does not exploit all
  information available to the adversary. The true privacy may be closer to
  the theoretical bound.
- **High noise regime:** When $\varepsilon < 1$, the signal is very weak and
  even optimal attacks cannot distinguish members from non-members reliably.

A small gap (audited $\approx$ theoretical) indicates:

- The attack is strong relative to the actual privacy level.
- The accounting is tight (PLD-based accounting typically is).

### When audited epsilon exceeds theoretical

This is a **red flag** indicating a likely implementation bug. Common causes:

- Gradient clipping applied incorrectly (e.g., clipping after summation)
- Noise scaled to wrong sensitivity
- Privacy accounting does not match the actual training procedure
- Data leakage outside the training loop (e.g., non-private evaluation on
  training data)

Investigate each component in isolation before concluding the accounting is
wrong.

## Comparison with shadow models

The classical approach to privacy auditing trains hundreds of "shadow models"
with and without a target example, then uses the distribution of model
behaviors to estimate membership. This is the gold standard for attack
strength but prohibitively expensive for large models.

Opaque's one-run method ([Steinke et al. 2023](https://arxiv.org/abs/2305.08846))
trades attack strength for efficiency: one training run gives a valid (though
potentially looser) epsilon bound. For most practical purposes — verifying
that your DP implementation is correct — the one-run approach is sufficient.

## Edge cases and limitations

- **Small datasets.** With fewer than 5000 examples, designating 1000 canaries
  removes 20%+ of the data, which can change training dynamics significantly.
  Use fewer canaries or a larger held-out pool.
- **Overfitting.** If the model memorizes the training set (common with small
  models and no regularization), the audit will show inflated epsilon. This
  does not mean DP is broken — it means the model has memorized, which is
  exactly what DP aims to prevent.
- **Non-convergence.** Auditing an unconverged model is meaningless: the loss
  scores will be noisy and the epsilon bound unreliable. Ensure the model has
  reached a reasonable training loss before evaluating.
- **Distributed training.** Run the audit on a single device with the same
  training configuration. DDP should produce identical results (same noise, same
  gradients), but running single-device simplifies debugging.

## Practical guidance

- **Use enough canaries.** 1000+ gives reliable results. 100 is too few
  for tight bounds.
- **Report confidence intervals.** Use `auc(confidence=0.95)` to quantify
  uncertainty.
- **Compare to theoretical epsilon.** The empirical bound should be lower.
  Use `summary(theoretical_epsilon=...)` for side-by-side display.
- **Audit multiple metrics.** AUC and beta at low alpha are complementary
  to epsilon.

## References

- Steinke, Nasr, Jagielski (2023). [Privacy Auditing with One (1) Training Run](https://arxiv.org/abs/2305.08846). NeurIPS 2023.
- Carlini et al. (2022). [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570).

## API reference

See [Auditing API Reference](../api/auditing.md) for complete function
signatures and return types.
