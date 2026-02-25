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

# 1. Setup
experiment = auditing.setup(dataset, num_canaries=1000, key=key(42))
train_data = dataset.select(experiment.train_indices(len(dataset)))

# 2. Train with DP-SGD on train_data ...

# 3. Evaluate
audit = auditing.evaluate(
    experiment, loss_fn, params,
    batch_argnums=(1,),
    dataset=dataset,
    collate_fn=collator,
    batch_unpack=lambda b: (b["input_ids"].to(device),),
)
print(audit.summary(delta=1e-5, theoretical_epsilon=target_eps))
```

## Integration with training

The auditing module is designed to slot into an existing DP-SGD training
loop with minimal changes. The key design principle: `auditing.evaluate`
accepts the same `loss_fn` and `batch_argnums` as `clipped_grad`, so the
same per-example loss function works for both training and auditing.

### Step 1: Setup before training

Call `auditing.setup` after tokenizing / preparing the dataset but before
creating the DataLoader. This selects canaries and flips coins:

```python
import opaque.auditing as auditing
from opaque.random import key

experiment = auditing.setup(train_dataset, num_canaries=1000, key=key(42))
full_dataset = train_dataset  # keep reference for scoring later

# Remove held-out canaries from training set
train_dataset = train_dataset.select(experiment.train_indices(len(train_dataset)))
```

For PyTorch `TensorDataset`, use `experiment.subset(dataset)` instead of
`dataset.select()`.

### Step 2: Train normally

No changes to the training loop. The dataset is already filtered.

### Step 3: Evaluate after training

Score all canaries (both in and out) against the trained model:

```python
audit = auditing.evaluate(
    experiment,
    per_example_loss_fn,    # same function used with clipped_grad
    trainable_params,       # trained parameters
    batch_argnums=(1,),     # same as clipped_grad
    dataset=full_dataset,   # the FULL dataset (before canary removal)
    collate_fn=data_collator,
    batch_unpack=lambda b: (b["input_ids"].to(device),),
)
print(audit.summary(delta=1e-5, theoretical_epsilon=target_epsilon))
```

The `batch_argnums`, `collate_fn`, and `batch_unpack` parameters handle
the mismatch between how a DataLoader yields batches and how the loss
function expects arguments:

- **`batch_argnums`**: Which positional arguments of `loss_fn` come from the
  dataset. `(1,)` means arg 1 is batched; `(1, 2)` means args 1 and 2 are
  batched. Same convention as `clipped_grad`.
- **`collate_fn`**: How to collate individual examples into a batch (e.g.,
  `DataCollatorForLanguageModeling` for HuggingFace).
- **`batch_unpack`**: How to extract tensors from a collated batch. For
  HuggingFace dict batches: `lambda b: (b["input_ids"].to(device),)`.

See [examples/train_causal_lm.py](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_causal_lm.py)
for a complete working example with the `--audit` flag.

## Epsilon estimation methods

### One-run (default for coin-flip experiments)

Likelihood-ratio test from Steinke et al. (2023). For each Pareto-optimal
threshold, tests both positive-only guesses and two-sided guesses, taking
the best result with Bonferroni correction. Only valid for coin-flip
experiments created via `auditing.setup`.

### Clopper-Pearson (default for direct construction)

Conservative binomial confidence intervals. Works with any in/out split,
including manual splits.

`epsilon_at()` automatically selects the method based on how the result was
created:

| Created via | Default method |
|---|---|
| `auditing.evaluate()` | `one_run` |
| `AuditResult(in_scores, out_scores)` | `clopper_pearson` |

You can override: `audit.epsilon_at(delta=1e-5, method="clopper_pearson")`.

## Attack metrics

```python
audit.auc()                  # ROC AUC (0.5 = random, 1.0 = perfect)
audit.beta_at(alpha=0.01)    # Type-II error at 1% FPR
audit.max_accuracy()         # Best threshold accuracy
```

AUC confidence intervals:

```python
ci = audit.auc(confidence=0.95, key=key(42))
print(f"AUC 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## Post-hoc auditing

If you already have membership scores:

```python
from opaque.auditing import AuditResult

audit = AuditResult(in_scores, out_scores)
print(audit.summary(delta=1e-5))
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
