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
from opaque.random import key

# 1. Partition: designate canaries and flip coins
cf = auditing.coin_flip(dataset, num_canaries=1000, key=key(42))
train_data = dataset.select(cf.train_indices(len(dataset)))

# 2. Train with DP-SGD on train_data ...

# 3. Score: compute membership scores for canaries
scores = auditing.loss_scores(
    loss_fn, trained_params,
    batch_argnums=(1,),
    dataset=dataset,
    indices=cf.canary_indices,
    collate_fn=data_collator,
    batch_unpack=lambda b: (b["input_ids"].to(device),),
)

# 4. Estimate: build the one-run estimate
estimate = auditing.one_run(scores, coin_flip=cf)
print(estimate.summary(delta=1e-5, theoretical_epsilon=target_eps))
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
    dataset=dataset,
    indices=cf.canary_indices,
    collate_fn=data_collator,
    batch_unpack=lambda b: (b["input_ids"].to(device),),
)
estimate = auditing.one_run(scores, coin_flip=cf)
print(estimate.summary(delta=1e-5, theoretical_epsilon=target_epsilon))
```

See [examples/train_causal_lm.py](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_causal_lm.py)
for a complete working example with the `--audit` flag.

### Parameter reference

These parameters on `loss_scores` bridge the gap between how a DataLoader
yields batches and how the loss function expects arguments:

- **`batch_argnums`**: Which positional arguments of `loss_fn` come from the
  dataset. `(1,)` means arg 1 is batched; `(1, 2)` means args 1 and 2 are
  batched. Same convention as `clipped_grad`.
- **`collate_fn`**: How to collate individual examples into a batch (e.g.,
  `DataCollatorForLanguageModeling` for HuggingFace).
- **`batch_unpack`**: How to extract tensors from a collated batch. For
  HuggingFace dict batches: `lambda b: (b["input_ids"].to(device),)`.

## Epsilon estimation

Epsilon is estimated using the one-run likelihood-ratio test from
Steinke et al. (2023). For each Pareto-optimal threshold, the test
tries both positive-only guesses and two-sided guesses, taking
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
