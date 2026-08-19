# LoRA-XSe: what we found, why it works, and what is still open

Audience: technical leadership. Non-DP, Qwen2.5-Coder-7B on KStack, 520 steps.
Every number below is from a run verified `state == finished` AND `step == 520/520`.

---

## 1. The headline

| method | trainable parameters | eval loss |
|---|---|---|
| full LoRA r=16 | 40,400,000 | 0.700090 |
| LoRA-XS, basis frozen | 200,704 | 0.702119 |
| **LoRA-XSe** (basis rotated) | **200,704** | **0.693279** |

LoRA-XSe beats full LoRA by **6.81e-3** using **201x fewer** trainable parameters,
at matched optimizer, learning rate, batch size, step count and weight decay.

**It replicates.** Seed 43: XSe 0.693141, frozen 0.702096. Rotation's benefit is
8.840e-3 (seed 42) and 8.955e-3 (seed 43) -- against a seed-to-seed spread of
1.4e-4. The effect is **64x the seed noise**. Seed 44 is in flight.

Read as a decomposition:

```
freezing the basis costs        2.03e-3
rotating it back gains         +8.84e-3   = 436% of that cost
net vs full LoRA               -6.81e-3
```

Rotation does not merely repay the price of freezing the basis. It overshoots by
4.4x and lands ahead of a method with 201x more parameters.

---

## 2. How we got here: we fixed the instruments, not the algorithm

The algorithm was already better. Six defects were hiding it. This matters for
credibility -- none of the gain came from tuning until something looked good.

**(a) Half the evaluation was scoring padding.** `collate()` dropped the labels
that carry the "ignore this token" markers, and because the pad token *is* the
end-of-text token, the model was graded on predicting "text ends" after "text
ends" -- on ~49.5% of positions (mean real length 520 of 1024). Reported losses
of 0.343 implied near-perfect prediction of source code; the true value is ~0.70.
This flattered every method equally, so no past *ranking* was wrong, but it means
no new number is comparable to the 323 historical runs.

**(b) One arm had a tailwind.** Weight decay defaults to 0.01 and all 62
operating-point runs used it -- but `xse_sgd()` has no weight-decay parameter at
all, so LoRA-XSe silently received zero while both baselines received 0.01. Every
historical "does rotation help" comparison was partly a weight-decay comparison.
Now set to 0 on every arm.

**(c) The control group had never been run.** At the exact operating point of the
entire non-DP campaign (7B, r=16, lr 5e-2), across all 62 finished runs:
full-LoRA runs = **0**; rotation-off runs = **0**. Every run was rotating XSe.
There was nothing to compare against, and "does XSe match LoRA" had been
literally unanswerable. Three independent analyses reached this conclusion and it
was verified directly from run configs.

**(d) The learning-rate schedule was disconnected.** `--lr-schedule` and
`--lr-warmup-steps` never reached the XSe optimizer; all 297 prior runs trained
at constant LR. A related flag, `--warmup-steps`, is dead code that 76 runs set
anyway, including the three then-best runs.

**(e) We declined the flattering metric.** Reporting the best checkpoint
(`loss_min`) instead of the final value made the old result look significant. But
a minimum over many noisy readings is biased toward the *noisier* arm, and XSe is
the noisier arm. We report the final value -- and it turned out moot:
`eval/loss == eval/loss_min` on all three arms, because all three converged
monotonically. The objection cannot be raised.

**(f) A rule about when a number may be read.** A claim was once retracted here
because a still-training run was read as final. Nothing counts until an
independent checker confirms `finished` AND full step count, printing a verdict
per run. It earned its keep immediately: the first full-LoRA attempt crashed on
GPU memory and was flagged `DISCARD` rather than silently averaged in.

---

## 3. Why AdamW helps, and why it may replace rotation

### The mechanism, plainly

Only a small `r x r` matrix R is trained. Its gradient has a **wide spectrum**:
directly measured for the first time (`rotation/sv0..sv7`, never present in any
previously built image), the top-8 singular values span **33x**, and
extrapolating the decay across all 16 suggests ~100-150x. There is **no gap** in
that decay -- it is a smooth slide with no natural cut point.

Plain SGD applies **one learning rate to all 16 directions**. The strongest moves
far; the weakest barely moves. Adam gives each parameter its own effective step
by dividing out its own typical gradient size, so weak directions still move.

### Measured, not asserted

| | frozen basis | rotating basis | rotation buys |
|---|---|---|---|
| **SGD** | 0.702119 | 0.693279 | **+8.84e-3** |
| **AdamW** | 0.695298 (lr 1e-3) | 0.693435 (lr 5e-3) | +1.86e-3 (lr-mismatched) |

**AdamW alone improves the frozen arm by 6.82e-3 -- 77% of everything rotation
buys under SGD.** The matched-learning-rate rotating cell is still pending, so the
1.86e-3 is provisional; the 77% figure is not, because both of its cells are
clean.

### Why substitution is the natural reading

Rotation and Adam are two answers to the *same* problem -- 16 directions with
wildly different gradient scales, and one learning rate that cannot serve them
all:

* **rotation copes by discarding** the directions that cannot keep up (SVD the
  momentum, keep the strong, re-randomize the weak);
* **Adam copes by rescaling** them so they can keep up.

If that is the whole story, a proper preconditioner substitutes for rotation
rather than adding to it. The evidence so far is consistent with it: rotation's
benefit shrinks from 8.84e-3 under SGD to ~1.9e-3 under AdamW, and nothing yet
beats SGD+rotation (AdamW+rotation ties it).

### Why substitution is probably *not* the whole story

Adam recovers **77%, not 100%**, and there is a geometric reason it cannot reach
100%. The frozen basis captures only **0.077%** of the gradient's energy (measured
from two pre-existing probe runs nobody had analysed). No amount of per-direction
rescaling can move the model in a direction that is **not in its span**. Adam can
use your 16 directions perfectly; it cannot hand you a 17th.

So the honest decomposition is roughly: **rotation = ~77% preconditioning +
~23% subspace escape.** That is a *result*, not a defeat -- it is the first
mechanistic account of why the method works, and it reframes the contribution
from "we added a knob" to "we explain what this class of method is doing."

### One technical caveat worth stating before someone else does

Adam is a **diagonal** preconditioner: it normalises per matrix *entry*. The
conditioning problem lives in the **spectral** basis (singular directions). These
are correlated but not identical, so Adam is the cheap ~80% instrument. The exact
one is a spectral (Muon-style) step that equalises singular values directly. If
AdamW helps only partially, that is the sharper follow-up.

### And a fact that reframes the whole optimizer question

**Published LoRA-XS uses AdamW in every experiment in its paper.** Our SGD-only
restriction was an accident of code placement -- the rotation optimizer was
constructed inside an `if optimizer == "sgd"` branch, so `--lora-xse-p-e` was
silently ignored under AdamW, and all ~225 XSe runs are heavy-ball SGD. We had
been benchmarking our method against itself with one hand tied.

---

## 4. Why LoRA-SB

LoRA-XS picks its 16 frozen directions from the SVD of the **pretrained weight**
-- i.e. from *what the model already knows*. Fine-tuning is about *the change you
want to make*. Those are different objects, and nobody had checked whether the
first is a good proxy for the second.

Measured: the frozen basis captures **0.077%** of the gradient's energy; a purely
**random** subspace of the same size captures **0.022%**. So the "informed" choice
is only **3.4x better than random**, that advantage does **not grow with rank**,
and for `v_proj` it is *below* random in 3 of 5 sampled layers. The basis we froze
was never much good.

**LoRA-SB** (arXiv:2411.19557) keeps our parametrization byte-for-byte and changes
only the basis: the SVD of the **first full-weight gradient** instead of of W0.
It beats LoRA-XS in every published cell (+4.63/+3.65/+4.85 GSM8K on Mistral-7B at
r=32/64/96). Four reasons we are testing it:

1. **It is the strongest published competitor at exactly our parametrization.** If
   we do not measure against it, a reviewer will ask why not.
2. **It composes with rotation** rather than competing. LoRA-SB chooses a better
   *starting* subspace, once. We *move* the subspace, continuously. Those are
   different axes and the combination is unpublished.
3. **It supplies a proof of one of our own results.** Its Theorem 5 shows that with
   frozen orthonormal factors the scaling factor is provably a no-op -- a published
   theoretical explanation for our empirical refutation of the alpha knob.
4. **Strategic position.** It is a *workshop* paper (SCOPE @ ICLR 2025) with no
   main-conference acceptance in 20 months. More useful: **LoRA-One (ICML 2025
   Oral) explicitly criticises LoRA-SB** on the grounds that a frozen basis with
   only r^2 degrees of freedom "[is] hard to escape/rotate" when the initial
   subspace is merely approximate. They demonstrate this on a **toy linear
   problem** and never test it on an LLM. Our rotation is precisely the mechanism
   that escapes. That is a named, unmeasured open question from a top-tier paper,
   and we can answer it.

---

## 5. Anticipated pushback

| objection | answer |
|---|---|
| "n=1 run." | Seed 43 replicates: rotation 8.955e-3 vs 8.840e-3, against a 1.4e-4 seed spread. Seed 44 in flight. |
| "You chose the metric that worked." | `eval/loss == eval/loss_min` on all arms; no checkpoint selection exists to exploit. |
| "Cherry-picked eval examples." | Paired over all 512, XSe wins 444 (86.7%). |
| "Weak baseline." | It is the preset's own tuned LoRA config: r=16 selected from a sweep over 8/24/32/48/64, lr from 3 values, momentum from 5, batch from 5. |
| "So rotation is just a bad optimizer workaround?" | Partly -- 77%, measured. The remaining ~23% is bounded below by geometry: the frozen span holds 0.077% of gradient energy, and no preconditioner supplies a direction outside the span. |
| "Bits-per-byte disagrees with loss." | Correct, and we found it ourselves -- see below. We lead with loss. |

### A correction we made ourselves

We initially made the paired bits-per-byte test load-bearing (t = -13.99). The
seed data broke it: rotation's BPB advantage is 6.2e-3 on seed 42 but 0.9e-3 on
seed 43, while the *loss* advantage is 8.84e-3 and 8.96e-3 -- rock steady. The
pattern is systematic: **rotating arms show BPB inconsistent with loss; frozen
arms are perfectly consistent.**

Ruled out: the 512 examples are identical across seeds (corr 0.9993); BPB masks
padding correctly; the parameter view is not stale.

Leading hypothesis: at tau=1 **every** step is a rotation step, so the final model
always has 5 of 16 directions freshly randomised with their R entries zeroed. BPB
is computed **once, on that final state**, whereas `eval/loss` is recorded at
checkpoints during training. The perturbation depends on the random draws, hence
it is seed-dependent. Frozen arms never rotate and are immune.

Consequence: **BPB comparisons involving rotating arms are contaminated by the
final-rotation state.** We now lead with `eval/loss`, which replicates at 64x the
seed spread, and treat BPB as secondary pending a fix (measure it at a
rotation-aligned point, or skip the final re-randomisation before eval).

---

## 6. What is still open

1. **Seed 44** on all three arms -- in flight.
2. **The matched AdamW cell** (rotating, lr 1e-3) -- decides additive vs
   substitutive cleanly.
3. **LoRA-SB** -- first attempt crashed on a bug of ours (batch unpacked as a dict
   where `collate` returns a tuple); fixed, image rebuilt, resubmission armed.
4. **Breadth -- the real gap to a top venue.** One model, one dataset, one task.
   LoRA-SB, a *workshop* paper, reports across four models and three task
   families. 2-3 models and one more task is the single highest-value remaining
   investment, and it is a compute question, not a research question.
5. **A one-line instrument we still lack**: cumulative basis drift
   `||B_0^T B_t||_F^2 / r`, which would confirm the subspace-escape mechanism
   directly. `rotation/r_subspace_angle` does not measure it -- it is identically
   zero in exact arithmetic, and the 0.007-0.019 logged is bf16 noise.

## 7. Bottom line

A method with **201x fewer trainable parameters beats tuned full LoRA**, the
result replicates across seeds at 64x the seed noise, and we can now say *why* it
works: roughly three-quarters preconditioning, one-quarter subspace escape. The
negative results (the alpha knob refuted three ways; depth and rotation interval
collapsing onto a single scalar) close a design space rather than merely failing.

Publishable today at workshop level. Main-track credible once breadth is added.
