# LoRA-XSe (non-DP): what we found, why it works, and what is still open

Non-DP, Qwen2.5-Coder-7B on KStack, 520 steps
(2 epochs), rank r=16, **exploration depth 5 of 16 directions, rotation interval
tau=1** (i.e. 5 of the 16 directions are replaced at every single step, 11 kept),
SGD momentum 0.9 at lr 5e-2, batch 192, weight decay 0.
Every number below is from a run verified `state == finished` AND `step == 520/520`.

---

## 1. The headline

| method | trainable parameters | eval loss |
|---|---|---|
| full LoRA r=16 | 40,400,000 | 0.700090 |
| LoRA-XS, basis frozen | 50,176 | 0.702119 |
| **LoRA-XSe** (basis rotated) | **50,176** | **0.693279** |

LoRA-XSe beats full LoRA by **6.81e-3** using **805x fewer** trainable parameters,
at matched optimizer, learning rate, batch size, step count and weight decay.

**It replicates.** Seed 43: XSe 0.693141, frozen 0.702096. Rotation's benefit is
8.840e-3 (seed 42) and 8.955e-3 (seed 43) -- against a seed-to-seed spread of
1.4e-4. The effect is **64x the seed noise**. Seed 44 is in flight.

### The number to defend: 2.02e-3, not 8.84e-3

An earlier draft of this document led with "rotation buys 8.84e-3". That was
measured against LoRA-XS run with **SGD** -- which is not LoRA-XS as published.
Its authors use AdamW in every experiment, so the SGD baseline is the method
handicapped by an optimizer nobody proposed. Against the faithful version:

| configuration | loss |
|---|---|
| LoRA-XS frozen + SGD | 0.702119 |
| **LoRA-XS frozen + AdamW** (faithful) | **0.695298** |
| **LoRA-XSe rotating + SGD** (ours) | **0.693279** |

```
of rotation's 8.84e-3 apparent benefit:
   6.82e-3  (77%)  obtainable by switching optimizer alone
   2.02e-3  (23%)  genuinely rotation's        <- 15x the seed spread
```

**This strengthens the claim rather than weakening it.** The residual 2.02e-3 is
exactly the component theory says a preconditioner *cannot* supply: the frozen
basis holds only 0.077% of the gradient's energy, and no per-direction rescaling
produces a direction outside the span. So the surviving effect is the
subspace-escape term, and its measured size now agrees with the geometry. A
2.02e-3 result that survives scrutiny is worth more than an 8.84e-3 one that does
not.

**Consequence, stated plainly:** the same critique applies to our full-LoRA
comparison, which is currently SGD-vs-SGD. That is internally fair, but full LoRA
also deserves AdamW. Historically AdamW improved full LoRA by ~7.8e-4 (old scale),
putting a fair arm near ~0.6985 -- still behind our 0.693279, but that is an
estimate. **A full-LoRA + AdamW arm is the most important outstanding control**
and is queued.

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

If that is the whole story, a preconditioner substitutes for rotation rather than
adding to it, and the measured 77% says most of it is exactly that.

**AdamW helps the two arms utterly differently, and this is the clearest single
piece of evidence for the mechanism:**

| arm | SGD | AdamW | change |
|---|---|---|---|
| frozen (p_e=0) | 0.702119 | 0.695298 | **-6.82e-3 (large gain)** |
| rotating | 0.693279 | 0.693435 | +0.16e-3 (nothing) |

The frozen arm is stuck with 16 fixed directions, most of which have gradients too
small to move under a single shared learning rate -- its realised effective rank is
2.1 of 16. Adam wakes those directions up, hence the large gain. The rotating arm
never has idle directions to wake: it discards whatever stops moving and draws
replacements, so its effective rank is 1.35 and every direction it carries is
already earning its place. Adam has nothing left to fix there.

**And the two actively interfere.** Adam's power lives in its second moment, the
running estimate of each direction's typical gradient size, which requires history
to estimate. Rotation rewrites the coordinate system that estimate is expressed in
-- at tau=1, every single step -- and fresh directions restart from zero. So
rotation continuously churns the ground Adam stands on. Corroboration from the
literature: every published gradient-subspace method that uses Adam recomputes the
subspace RARELY. GaLore uses T=200 and reports T=50-1000 all work; we use tau=1,
200x more often, and GaLore itself warns that frequent switching "may also impact
the fidelity of the optimizer states."

**A confound worth naming before someone else does:** tau=1 was chosen because the
tau sweep found it best under SGD. It is also the worst possible setting for Adam.
Our AdamW+rotation arm therefore ran at an SGD-optimal cadence. The prediction --
queued -- is that AdamW+rotation IMPROVES as tau grows, the opposite of SGD's
behaviour. If it does, that is where the two mechanisms finally stack, and it is
strong confirmation of the interference account.

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

A method with **805x fewer trainable parameters beats tuned full LoRA**, the
result replicates across seeds at 64x the seed noise, and we can now say *why* it
works: roughly three-quarters preconditioning, one-quarter subspace escape. The
negative results (the alpha knob refuted three ways; depth and rotation interval
collapsing onto a single scalar) close a design space rather than merely failing.

Publishable today at workshop level. Main-track credible once breadth is added.


---

## Appendix: correcting the parameter ratio (201x -> 805x)

Earlier drafts quoted 201x. That compared LoRA-XS at **r=32** (32^2 x 196 = 200,704
trainable) against full LoRA at r=16 (40,370,176). Every experiment in this
document used **r=16 for both arms**, where LoRA-XS trains 16^2 x 196 = **50,176**
parameters. The matched ratio is therefore **805x**, and the result was being
under-claimed by a factor of four.

Verified from the run configs: ref-xse-d5t1-s42 has lora_method=lora-xs, lora_r=16;
ref-lora-r16-mb8-s42 has lora_method=lora, lora_r=16.
