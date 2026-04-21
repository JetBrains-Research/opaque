# Privacy Auditing

Privacy accounting gives theoretical upper bounds on privacy loss.
Privacy auditing gives empirical lower bounds by running a membership
inference attack. If the audited epsilon exceeds the theoretical epsilon,
there is likely a bug in the implementation.

## How it works

Opaque implements one-run auditing from
[Steinke, Nasr, Jagielski (2023)](https://arxiv.org/abs/2305.08846):

1. **Designate canaries.** Randomly select `m` examples as canaries.
2. **Flip coins.** For each canary, include or exclude it from training
   with probability 0.5.
3. **Train once.** Train on the resulting subset. Non-canary data is always
   included.
4. **Score.** Compute a membership score for each canary (negative loss by
   default — lower loss means more likely a member).
5. **Test.** Use a likelihood-ratio test to bound epsilon from the scores
   and coin flips.

Only one training run is needed, unlike shadow-model approaches.

## Quick start

```python
import opaque.auditing as auditing
from opaque.core.random import key
from torch.utils.data import DataLoader, Subset

# 1. Partition: designate canaries and flip coins
cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
train_data = dataset.select(cf.train_indices(len(dataset)))

# 2. Train with DP-SGD on train_data ...

# 3. Score: compute membership scores for canaries
def canary_collate(examples):
    batch = data_collator(examples)
    return (batch["input_ids"].to(device),)

canary_loader = DataLoader(
    Subset(dataset, cf.canary_indices.tolist()),
    batch_size=32, collate_fn=canary_collate,
)
scores = auditing.loss_scores(
    loss_fn, trained_params,
    batch_argnums=(1,),
    dataloader=canary_loader,
)

# 4. Estimate: build the one-run estimate
estimate = auditing.one_run(scores, coin_flip=cf)
print(f"ε (empirical): {estimate.epsilon_at(delta=1e-5):.4f}")
print(f"AUC: {estimate.auc():.4f}")
```

## Integration with training

The three-step API separates concerns: **partition** (before training),
**score** (after training), **estimate** (compute metrics).

### Step 1: Partition before training

Call `auditing.coin_flip` after preparing the dataset:

```python
cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))

# Remove held-out canaries from training set
train_dataset = dataset.select(cf.train_indices(len(dataset)))
```

### Step 2: Train normally

No changes to the training loop. The dataset is already filtered.

### Step 3: Score and estimate after training

```python
scores = auditing.loss_scores(
    per_example_loss_fn, trained_params,
    batch_argnums=(1,),
    dataloader=canary_loader,
)
estimate = auditing.one_run(scores, coin_flip=cf)
print(f"ε (empirical): {estimate.epsilon_at(delta=1e-5):.4f}")
```

See [examples/train_causal_lm.py](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_causal_lm.py)
for a complete working example with the `--audit` flag.

### Parameter reference

- **`batch_argnums`**: Which positional arguments of `loss_fn` come from the
  dataloader. `(1,)` means arg 1 is batched; `(1, 2)` means args 1 and 2 are
  batched. Same convention as `clipped_grad`.
- **`dataloader`**: Any iterable yielding batches. Each batch should be a
  tensor (single `batch_argnums`) or a tuple of tensors (multiple
  `batch_argnums`). Use a custom `collate_fn` on the DataLoader to handle
  dict-style batches (e.g., HuggingFace).
- **`reference_scores`**: Baseline scores from an untrained model. When
  provided, returned scores are `scores - reference_scores` (loss reduction).

## Epsilon estimation

Epsilon is estimated using the one-run likelihood-ratio test from
Steinke et al. (2023). For each Pareto-optimal threshold, the test
tries positive-only, negative-only, and two-sided guesses, taking
the best result with Bonferroni correction.

```python
estimate.epsilon_at(delta=1e-5)                       # default significance=0.05
estimate.epsilon_at(delta=1e-5, significance=0.01)    # stricter confidence
estimate.epsilon_at(delta=1e-5, threshold=4.0)        # specific threshold
```

## Attack metrics

```python
estimate.auc()                  # ROC AUC (0.5 = random, 1.0 = perfect)
estimate.beta_at(alpha=0.01)    # Type-II error at 1% FPR
```

AUC confidence intervals:

```python
ci = estimate.auc(confidence=0.95, key=key(42))
print(f"AUC 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## Number of canaries

More canaries = tighter bounds:

| Canaries | Use case |
|---|---|
| 100 | Debugging only, very noisy |
| 500 | Detecting large implementation bugs |
| 1000+ | Recommended for reliable bounds |
| 5000+ | Needed for small-epsilon regimes |

## Interpreting results

The audited epsilon should be **below** the theoretical epsilon:

```
  ε (one-run):          1.50
  ε (theoretical):      3.00    ← expected: gap exists
```

**Gap is expected.** The audit runs a suboptimal attack; the theoretical
bound is a worst-case upper bound. A large gap is normal, especially in
high-noise regimes.

**Audited > theoretical is a red flag.** Investigate:

- Gradient clipping applied incorrectly
- Noise scaled to wrong sensitivity
- Privacy accounting doesn't match actual training procedure
- Data leakage outside the private training loop

## References

- Steinke, Nasr, Jagielski (2023). [Privacy Auditing with One (1) Training Run](https://arxiv.org/abs/2305.08846). NeurIPS 2023.
- Carlini et al. (2022). [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570).

## API reference

See [Auditing API Reference](../api/auditing.md) for complete function
signatures.
