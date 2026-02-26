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

# 1. Setup: designate canaries and configure scoring
audit_state = auditing.setup(
    dataset, num_canaries=1000, key=key(42),
    batch_argnums=(1,),
    collate_fn=data_collator,
    batch_unpack=lambda b: (b["input_ids"].to(device),),
)
train_data = dataset.select(audit_state.train_indices)

# 2. Train with DP-SGD on train_data ...

# 3. Evaluate: just pass loss_fn and trained params
result = audit_state.evaluate(loss_fn, trained_params)
print(result.summary(delta=1e-5, theoretical_epsilon=target_eps))
```

## Integration with training

The key design principle: configure scoring once at `setup()` time, then
`evaluate()` is a one-liner after training.

### Step 1: Setup before training

Call `auditing.setup` after preparing the dataset. Pass the same
`batch_argnums` you use with `clipped_grad`, plus `collate_fn` and
`batch_unpack` for HuggingFace dict batches:

```python
audit_state = auditing.setup(
    train_dataset,
    num_canaries=1000,
    key=key(42),
    # Same batch_argnums as clipped_grad:
    batch_argnums=(1,),
    # Same collate_fn as your DataLoader:
    collate_fn=data_collator,
    # How to extract tensors from a collated batch:
    batch_unpack=lambda b: (b["input_ids"].to(device),),
)

# Remove held-out canaries from training set
train_dataset = train_dataset.select(audit_state.train_indices)
```

The `OneRunEstimator` remembers both the full dataset and the scoring config.
For PyTorch `TensorDataset`, you can omit `collate_fn` and `batch_unpack`.

### Step 2: Train normally

No changes to the training loop. The dataset is already filtered.

### Step 3: Evaluate after training

```python
result = audit_state.evaluate(per_example_loss_fn, trained_params)
print(result.summary(delta=1e-5, theoretical_epsilon=target_epsilon))
```

That's it. The dataset, `batch_argnums`, `collate_fn`, and `batch_unpack`
are all stored in the `OneRunEstimator` from step 1.

See [examples/train_causal_lm.py](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_causal_lm.py)
for a complete working example with the `--audit` flag.

### Parameter reference

These parameters bridge the gap between how a DataLoader yields batches
and how the loss function expects arguments:

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
audit.epsilon_at(delta=1e-5)                 # convenience (calls epsilon_one_run)
audit.epsilon_one_run(significance=0.05)     # explicit
```

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
