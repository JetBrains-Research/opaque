# Batch 1 — LoRA-XSe beats full LoRA at 201x fewer parameters (non-DP, 7B)

**Status: FINAL.** All three arms verified `state == "finished"` AND `step == 520/520`
by `campaign_logs/queue/wait_runs.py`, which prints an explicit readability verdict
per run. A fourth run (`ref-lora-r16-s42`) is marked DISCARD and is not used.

Image `david-stan-zenml-training-fec7ae2`. Model Qwen2.5-Coder-7B, dataset KStack,
preset `qwen-coder-kstack-lora`, SGD momentum 0.9, lr 5e-2, batch 192, 2 epochs
(520 steps), r=16, seed 42, `--weight-decay 0`, `--eval-batch-size 16`, `--eval-bpb`.

## Headline

| arm | run | trainable params | eval loss | BPB |
|---|---|---|---|---|
| full LoRA r=16 | `ref-lora-r16-mb8-s42` | 40.4M | 0.7000900 | 0.269005 |
| frozen LoRA-XS (`p_e=0`) | `ref-xs-norot-s42` | 200,704 | 0.7021185 | 0.271946 |
| **LoRA-XSe** (d=5, tau=1) | `ref-xse-d5t1-s42` | **200,704** | **0.6932789** | **0.265767** |

**LoRA-XSe beats full LoRA by 6.81e-3 eval loss (3.24e-3 BPB) with 201x fewer
trainable parameters**, at matched optimizer, data, steps and weight decay.

`eval/loss == eval/loss_min` for all three arms. Every arm converged monotonically,
so no minimum-over-checkpoints selection is involved anywhere in this result. That
matters: `loss_min` is a biased estimator that favours the noisier arm, and XSe is
the noisier arm, so any result resting on it would be correctly attacked.

## Decomposition

```
frozen-basis cost   (LoRA - frozen) = 2.03e-3      what freezing B,A costs
rotation value      (frozen - XSe)  = 8.84e-3      what rotation buys
                                    = 436% of the frozen-basis cost
net advantage over full LoRA        = 6.81e-3
```

Rotation does not merely recover the penalty for freezing the basis -- it exceeds
it 4.4x and lands ahead of full LoRA. This is stronger than the eps=3 history,
where the same ratio was ~103%.

## Paired per-example BPB — the statistically load-bearing test

BPB is logged per example, so all three arms are scored on the SAME 512 eval
examples and the comparison is paired. This is the first image able to compute it.

| comparison | mean paired diff | 95% CI (20k bootstrap) | paired t | examples won |
|---|---|---|---|---|
| **XSe vs full LoRA** | **-0.003238** | [-0.003697, -0.002795] | **-13.99** | **444/512 (86.7%)** |
| XSe vs frozen | -0.006179 | [-0.007016, -0.005382] | -14.76 | 455/512 (88.9%) |
| full LoRA vs frozen | -0.002941 | [-0.003592, -0.002325] | -9.04 | 318/512 (62.1%) |

The unpaired SEMs are ~0.0073 per arm -- LARGER than the effect, so an unpaired
comparison would have called this noise. Pairing tightens the standard error 17-32x.

**Scope of the claim, stated precisely.** These tests establish that the differences
are not eval-sampling noise, at n=1 RUN per arm. They say nothing about
seed-to-seed variance, which was historically ~5 noise-floor units for XSe. Two
further seeds per arm are required before the result is robust rather than merely
decisive on this run. Do not report it as significant across runs until then.

## Mechanism telemetry (first direct observations in the project)

`rotation/sv0..sv7` has existed in the tree since 2456a38f but had never been in a
built image, so the momentum spectrum had never been observed -- only inferred.

| quantity | frozen | LoRA-XSe |
|---|---|---|
| `xs/r_effective_rank` | 2.0997 | **1.3500** |
| `xs/r_condition` | 2,578.7 | 4.12e9 |
| `xs/r_norm` | 0.025336 | 0.078136 |
| `rotation/promotion_count` | - | 0.1378 (2.8% of 5 draws) |

Momentum spectrum, XSe arm:
`sv0..sv7 = 0.009207, 0.003482, 0.001758, 0.001011, 0.0006079, 0.0004344, 0.0003445, 0.0002793`
=> **sv0/sv7 = 33.0x**, decelerating; extrapolating to sv15 suggests ~100-150x
overall. Two corrections follow from this:

1. The previously circulated figure of ~307x was INFERRED from `energy_ratio` and
   `top_ratio_median`, and is too large. The qualitative premise survives -- a
   large spread with no spectral gap -- but the magnitude was overstated.
2. **The winning arm uses FEWER effective directions, not more** (1.35 vs 2.10) and
   is far more concentrated. So "more usable directions" is not the objective, and
   any hypothesis predicting that AdamW helps by RAISING effective rank is not a
   valid test of the mechanism. Retracted.

Point 2 also sharpens the substitution risk for the AdamW work: if SGD can only
really move one direction, rotation's contribution may be *choosing which one*, in
which case a preconditioner that makes several directions usable could substitute
for rotation rather than add to it. The 2x2 in batch 2 is the test.

## Provenance and known-clean confounds

* `--weight-decay 0` on every arm. At the default 0.01 the non-XSe branch applies
  decay (`train_causal_lm.py:1975`) while `xse_sgd()` takes no weight_decay
  argument at all, so a default-config comparison is partly a weight-decay
  comparison. Every historical LoRA-vs-XSe contrast carries that confound.
* `--eval-batch-size 16` pinned on all three. The LoRA arm needed
  `--microbatch-size 8` to clear a CUDA OOM in the vocab-projection kernel
  (`linear_cross_entropy.py:857`, 16.24 GiB); since `eval_batch_size` defaults to
  `microbatch_size`, leaving it implicit would have moved that arm's eval batch to
  8. Microbatching is gradient accumulation, so the training math is unchanged.
* Absolute losses are ~2x every pre-`2456a38f` number because the padding-mask fix
  removed `<eos>`-after-`<eos>` scoring on ~49.5% of positions. NOT comparable to
  the 323-run historical corpus. All three arms here are on the same scale.
* `downstream/*`, HumanEval and MBPP remain invalid (stale-basis defect) and are
  off by default; none are used above.

## Still open

* Seeds (43, 44) on all three arms -- converts "decisive on this run" to "decisive".
* Batch 2: the AdamW 2x2 and the LoRA-SB gradient basis (blocked on VPN).
* One unmeasured quantity that would settle the cage-escape mechanism directly:
  cumulative basis drift `||B_0^T B_t||_F^2 / r`. `rotation/r_subspace_angle` does
  NOT measure it -- it is identically zero in exact arithmetic, and the 0.007-0.019
  logged is bf16 drift.
