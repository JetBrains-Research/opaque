# Synthesis: what to do next, and the mechanism reframe

Consolidates three parallel research passes (utility headroom / signal-processing / evaluation) plus my
own verification. **Every claim below is marked VERIFIED (I checked it at source or in the data),
REPORTED (agent finding, not independently checked), or CORRECTED.**
Companion to `docs/renyi-alpha-theory-final.md`. Date 2026-08-17.

---

## 0. The headline: the mechanism is DISCARDING noise, not DISCOVERING directions

Three independent lines converge, and this reframes the paper's central claim.

**(a) Random exploration cannot discover a useful direction. VERIFIED — arithmetic.**
`random_orthogonal_complement` draws `B_explore` uniformly in the orthogonal complement of the current
span, in ambient dimension `d_out` (3584 for q/k/v/o, 18944 for gate/up). A random unit vector's
expected squared overlap with any *fixed* target is `1/d_out`, and the best of N draws grows only as
`log N / d_out`.

| module | r_e | τ | rotations | draws | best sq-overlap | rotations *needed* | coverage |
|---|---|---|---|---|---|---|---|
| q/o (d=3584) | 5 | 5 | 104 | 520 | 0.0017 | ~717 | **15 %** |
| q/o | 13 | 5 | 104 | 1352 | 0.0020 | ~276 | 38 % |
| gate/up (d=18944) | 5 | 5 | 104 | 520 | 0.0003 | ~3789 | **3 %** |

> **The explore block covers 3–15 % of the measure needed to find one genuinely new ambient direction,
> and raising τ makes it worse.** So whatever `p_e > 0` buys, it is not discovery.

**(b) The benefit scales linearly with injected noise. VERIFIED — and this CORRECTS the agent.**
The agent reported rotation buying 0.0012 at ε=3 vs 0.0157 at ε=1 ("13×"). The actual `cmp-` numbers:

| | frozen basis | rotating | rotation buys |
|---|---|---|---|
| ε=3 (n=3/3) | 0.351958 | 0.345273 | **6.685e-3** |
| ε=1 (n=3/2) | 0.354545 | 0.345593 | **8.952e-3** |

Ratio **1.34×** — not 13×. **But the correction strengthens the argument:** the injected-noise ratio is
`0.00405/0.00307 = 1.32×`. So the benefit scales with noise magnitude to **within 2 %**. A discard
mechanism predicts benefit ∝ noise; a discovery mechanism predicts no such relation. 13× would have been
*unexplained*; 1.34× is the prediction landing.

**(c) Rotation demonstrably suppresses noise accumulation in R. REPORTED.**
At matched clipping norm, going ε=3 → ε=1: ‖R‖ grows **+90 % frozen** vs **+24 % rotating**, and
`r_eff(R)` is 15.7 (frozen) vs 5.9 (rotating) at ε=3.

**Why this is a better story than "exploration".** It explains, in one mechanism, three things the
exploration framing does not: the **flat depth curve** (once you discard enough noise, how much more you
discard barely matters), the **ε-monotonicity** (more noise ⇒ more to discard), and why **per-matrix
adaptivity is worthless** (noise is isotropic, so there is nothing matrix-specific to adapt to).

### 0.1 The decisive experiment — do this one first

**Reset-in-place.** Keep the same basis directions, but zero the `R` entries and momentum of the slots
that rotation would have replaced. This discards accumulated noise **without changing the subspace**.

- If it **matches** full rotation ⇒ the mechanism is discarding. The paper's framing changes, and every
  "which directions to explore" question closes permanently.
- If it **recovers little** ⇒ the subspace change matters after all, and exploration survives.

~15 lines. Run at **ε=1**, 5 seeds, endpoint `eval/loss`. Resolvable: the effect under test is
**8.95e-3 against a ~6.5e-5 floor = 138×**, the largest signal-to-floor ratio in the project.
Either outcome is publishable, and a reviewer will ask.

---

## 1. Two code defects — VERIFIED at source. Neither retracts any verdict.

> **STATUS 2026-08-17: two fixes now applied in the working tree (not built, not committed).**
>
> 1. **Padding mask** — `per_example_loss_fn` (`train_causal_lm.py:~1535`) now builds
>    `labels = input_ids.masked_fill(input_ids == pad_id, -100)`. Reconstructed in-function rather
>    than threaded through the signature, because the vmap'd `clipped_grad` /`auto_clipped_grad` /
>    `adaptive_clipped_grad` call sites bind it as `f(trainable, batch_elem)` and an extra argument
>    would ripple into the `opaque` library. **Verified byte-identical to
>    `DataCollatorForLanguageModeling`'s own labels** (`torch.equal` → True on a padded batch).
>    ⚠️ **BREAKING for comparability.** This changes both the training loss and the eval metric.
>    Absolute losses will roughly double (real CE ~0.68 vs the diluted ~0.34) and **will not be
>    comparable to any of the 297 existing runs.** A fresh baseline is required after the rebuild.
> 2. **LR schedule** — `xse_sgd(lr=lr_for_opt, …)` instead of `lr=args.learning_rate`
>    (`train_causal_lm.py:~1934`). **Non-breaking:** `--lr-schedule` still defaults to `none`, so
>    behaviour is unchanged unless the flag is set. It simply starts working.
>
> Neither affects any in-flight run — those execute the prebuilt image `…-7e97389`, and local edits
> take effect only on the next `build_and_push.sh`.

**(1) The downstream path uses a STALE basis.** `train_causal_lm.py:2786` takes
`peft_model = model._module …` and calls `save_pretrained`, then feeds `peft_model` to HumanEval/MBPP.
That is the **module**, not the functional model. Rotation writes `A_new = A.clone()` /
`B_new = B.clone()` into the `new_frozen` **dict** (`xse.py:777-782`) — new storage, module untouched.
⇒ downstream eval and saved adapters see `(A_init, B_init, R_final)`.
**Every `downstream/*` number for the ~225 `p_e>0` runs, and every saved adapter, is invalid.**
Independently REPORTED: the harness also scores the *base* model at 0.061 pass@1 (published ≈0.61), so
there are two stacked bugs. **Fix or drop all downstream claims.**

**(2) Padding is scored in `eval/loss`.** `collate` returns only `(batch["input_ids"],)` — it
**discards** `batch["labels"]`, which is exactly where `DataCollatorForLanguageModeling` puts the `-100`
pad mask (`train_causal_lm.py:1439-1441`). `per_example_loss_fn` then passes `labels=input_ids`, the
padded tensor, and `pad_token = eos_token`, so the model is scored on predicting `<eos>` after `<eos>`.

**The fraction is now VERIFIED independently** (I re-tokenised 300 KStack files with the Qwen2.5-Coder
tokenizer at `max_length=1024` and replayed `DataCollatorForLanguageModeling` batching at
`eval_batch_size=16`):

| quantity | my reproduction | agent |
|---|---|---|
| mean real tokens | **520.1** (median 485.5) | 508 |
| files hitting the 1024 cap | 24.0 % | — |
| **pad share of scored positions** | **49.5 %** | 50.4 % |

(Padding to the batch max rather than to 1024 makes almost no difference — 49.5 % vs 49.2 % — because
with a 24 % cap rate essentially every batch of 16 contains at least one full-length file.)

**And the perplexity arithmetic closes exactly.** Reported `eval/loss` 0.3435 ⇒ perplexity **1.410**,
which is not credible for source code. Removing ~49.5 % near-zero pad positions gives real CE **0.681**
⇒ perplexity **1.98** — entirely plausible for a fine-tuned 7B code model on Kotlin. The defect
quantitatively resolves an implausibility that should have been challenged long ago.
**SNR-neutral — pad positions carry no between-run variance, so every comparison stands and no verdict
moves.** But the absolute numbers are not defensible in a paper, and part of the headline "2.617 → 0.344"
is the model learning to emit padding. `eval_bpb` already masks correctly (`valid = (tgt != pad_id)`).

**(3) `eval/loss` itself is CORRECT — VERIFIED, and this contradicts one agent's framing.**
`main()` opens at line 1109; the `merged_params` closure (1531) and the `frozen_params` reassignment
(2168) are both inside it with no intervening `def`, so the closure reads the rebound value.
`eval_loss → per_example_loss_fn → fmodel(merged_params(trainable), …)` **does** see the rotated basis.
**Everything the α, depth, floor and margin analyses rest on is computed correctly.**

**(4) Repo-level leakage. REPORTED, unverified.** 808/1000 eval files share a repo owner with training
data; 544/850 distinct repos leak. Files are disjoint (`take`/`skip`); repos are not.

---

## 2. Corrections to my own earlier claims

| I said | Correction | Source |
|---|---|---|
| "eval loss wobbles ±1.9e-4 per 10 steps" | The curve is **monotone**; `|loss(t)−loss(t−10)|` measures the **trend**, not jitter. True residuals are 1.3e-5–5e-5. | VERIFIED |
| "train to convergence to shrink the floor" | Buys only **1.6×** (7.0e-5 → 4.3e-5) while effects decay *faster* (`re9−re13` shrinks 45× by step 390). | REPORTED |
| "the noise floor is a scalar (6.5e-5)" | Better model: `σ = √(c₀² + (k·|dL/ds|)²)`, c₀ = 4.3e-5 irreducible, k ≈ 5.6 steps. **GPU non-determinism alone is 5.1e-5 = 62 % of the floor variance.** | REPORTED |
| §10.1d: "adaptive vs uniform at m=8 is category-stable" | It **flips** (0.89× → 1.13×) under the non-determinism-only floor. Only the depth penalty and α are stable. | REPORTED |
| §8.2: the threshold rule is the constructive fix | MP thresholding **has been run** at 2 epochs (c=1/1.5/2 → 0.345301/0.345270/0.344875, plus scheduled and BBP variants), all inside the vanilla range 0.345030–0.345641. **Null.** Not identical to my proposal (it truncates *before* the entropy step rather than replacing the count rule) but close enough to lower confidence. | VERIFIED |
| "the 1.01e-3 τ effect" | **One** matched pair exists in all 297 runs, and XSe's own p90 eval excursion is 1.09e-3. The effect is at jitter scale, n=1. | VERIFIED |

---

## 3. Two risks to the paper as written — both already on disk, zero GPU

**(a) A properly-tuned full-LoRA baseline beats every LoRA-XSe run by ~8e-4. VERIFIED.**
`lora-r16-lr1e-3-bs192-adamw`, finished, 520 steps, ε=3, bs=192, **AdamW lr 1e-3 → 0.344165**.
Best XSe: 0.344718 single / ~0.345024 as a clean group. The paper's comparator is *SGD at lr 5e-2*
(0.34494–0.34556) and REPORTED as still descending at −4.7e-4/100 steps, i.e. undertrained.

**(b) "Matches full LoRA" is 7B-specific. VERIFIED.**

| model | full LoRA r=16 | LoRA-XSe r=32 | frozen XS r=32 | LoRA − XSe |
|---|---|---|---|---|
| 7B | 0.344941 | 0.345306 | 0.352025 | −3.7e-4 (tie) |
| **14B** | **0.322032** | **0.324728** | 0.331697 | **−2.70e-3** |

All three 14B runs finished at 520 steps, matched on ε, lr, batch, optimizer. **n=1 per cell.**
The *other* half is scale-robust: XSe beats the frozen basis by 6.97e-3 at 14B vs 6.68e-3 at 7B — which
is exactly what §0's discard model predicts, since it is the noise-suppression half.
**Either scope the parity claim to 7B or verify with seeds.**

---

## 4. Verified positives worth taking

- **α is identifiable under DP and still inert. VERIFIED.** At r=32 under DP the α→depth map spans
  **10.8 → 23.3 = 12.6 slots**, versus ≤0.8 non-DP. This is the "escape the null space" experiment of
  §8.1 #2 — already run, never analysed. It upgrades the negative from *α unidentifiable* to
  **α identifiable and still inert**, the strongest form. Caveat: the agent's quoted 4.0e-4 loss span is
  a **subset**; the full r=32 DP spread is wider and contains at least one excursion, so this needs a
  proper paired analysis before publication.
- **The age-bias law is exact. REPORTED, zero fitted parameters.** Monte-Carlo of the real rotation
  (12 seeds, 1000 steps) shows freshly planted directions are under-retained by exactly
  `√(1−β^{2τ})`: predicted/measured 0.6845/0.679, 0.8070/0.825, 0.9372/0.947, 0.9974/1.002 at
  τ = 3/5/10/25 — max error 2.2 % across an 8× range, and the deficit vanishes at τ=25.
  The Adam-style correction `W_ij = 1/(1−β^min(aᵢ,aⱼ))` removes it at every τ.
  **But §0 lowers its priority:** promoting newcomers is not where the value is.
- **The eval/rotation phase question is answered: null. REPORTED.** Joint F(4,21) = 0.090, p = 0.985 on
  the landed `phase-eval7` run; establishes |phase| ≤ 1.0e-4. Free second test: 76 cached runs already
  use interval 3, which cycles phase against `eval_steps=10`.
- **BPB cannot improve resolution. REPORTED, arithmetic verified by the agent.** One tokenizer + fixed
  set ⇒ `BPB = 0.3145 × nats/token` exactly, so identical SNR. Adopt for interpretability only.
- **No capability metric can resolve our effects. REPORTED.** Resolving 3.46e-3 nats needs ~275k tasks
  at 80 % power; Kotlin_HumanEval has 161. MBPP+ between-arm variance is entirely binomial. Run for
  external validity, and say so.
- **Paired seeds / CRN buy nothing. REPORTED.** Contrast replication error is 1.73× the level error —
  *above* the 1.41× expected under independence. Retires a standing plan item.

---

## 5. Wavelets and Fourier: a well-argued negative, with one exception

Ranked last by the signal-processing pass, with an explicit "don't". The SVD is already the natural
spectral decomposition of a matrix, and a wavelet decomposition of the gradient time series reduces to
the persistence estimator with extra machinery. The loss-spectral analysis is dominated by data order
(established earlier: the phase pattern moves with the seed).

**The exception, and it was productive:** the *time-domain filter* analysis — treating the momentum as a
single-pole IIR filter with memory `1/(1−β)` = 10 steps sampled every τ=5 — produced the exact
age-bias law in §4 and the τ recommendation. So the filtering *framing* paid off; the wavelet
*machinery* does not.

---

## 5.5 LANDED: the rotation INTERVAL is the largest utility lever in the project. VERIFIED.

The 2×2 disentangler completed (all arms `state=finished`, 520/520). It answered its own question and
surfaced something bigger.

**(a) The depth-effect halving is BOTH training and rotation count, ~55/45.**

| condition | rotations | steps | depth 1 | depth 5 | gap |
|---|---|---|---|---|---|
| A: 1 ep, τ=5 | 52 | 260 | 0.347000 | 0.344290 | 2.71e-3 |
| B: 2 ep, τ=5 | 104 | 520 | 0.344421 | 0.343068 | **1.35e-3** |
| C: 2 ep, τ=10 | **52** | 520 | 0.346412 | 0.344445 | 1.97e-3 |

- **A→C** (training doubles, rotations fixed): **−7.43e-4** — the *training* component, **55 %**
- **C→B** (rotations double, training fixed): **−6.14e-4** — the *rotation-count* component, **45 %**

Neither hypothesis alone explains the halving. So the depth rule is partly a convergence-rate artifact
*and* partly a total-exploration effect, and it should be stated with both caveats.

**(b) The big one: rotating more often is worth 1.4–2.0e-3.** At matched training (2 epochs), the only
difference between B and C is τ:

| depth | τ=5 | τ=10 | τ=5 advantage | vs floor |
|---|---|---|---|---|
| 1 | 0.344421 | 0.346412 | **1.99e-3** | **30.6×** |
| 5 | 0.343068 | 0.344445 | **1.38e-3** | **21.2×** |

**2/2 matched pairs, same direction, 21–31× the floor.** And unlike the earlier τ=3 vs τ=5 pair, this
comparison has **no phase confound at all** — τ=5 and τ=10 are *both* fully phase-aligned with
`eval_steps=10`, and the landed `phase-eval7` run independently bounds any phase effect at ≤1.0e-4,
two orders below the effect.

Combining with the τ=3 vs τ=5 pair (τ=3 better by 1.01e-3, §2):

> **τ = 3 < 5 < 10, monotone, ~1–2e-3 per doubling.**

**This dwarfs every other lever measured:** the LR schedule's predicted 1.5–5e-4, rank-down's ≤4e-4
cost, and α's *entire* reach of 2.8e-5. And τ has taken **exactly two values in all 297 runs** and has
never been scheduled. It is the single most under-explored axis in the project.

**Next:** sweep **τ ∈ {1, 2, 3} at 2 epochs**, with `--eval-steps 7` so no candidate τ is phase-locked.
Rotation is essentially free in wall-clock, so τ=1 costs nothing extra. A turning point is expected —
at τ=1 the post-rotation gradient-norm penalty (REPORTED: +9.6 % on the following step) would apply to
100 % of steps instead of 33 %. **Falsify:** if τ=1 and τ=2 both land within 1.5e-4 of τ=3, the axis is
saturated and stop.

**Caveat worth stating:** the τ effect and the "discard not discover" mechanism (§0) are consistent —
if rotation works by discarding accumulated noise, discarding *more often* should help, and the ambient
dimension argument says nothing is lost by not exploring longer. τ is the dose of the actual mechanism.

## 6. Ranked plan

| # | action | GPU | why |
|---|---|---|---|
| **1** | **τ sweep: τ ∈ {1,2,3} at 2 epochs, `--eval-steps 7`** (§5.5) | 3–5 runs | **largest measured lever: 1.4–2.0e-3 = 21–31× floor, 2/2 matched pairs, no phase confound.** Only 2 values tried in 297 runs |
| **2** | **Reset-in-place discard test** (§0.1) | 5 runs | decides the paper's mechanism; 138× floor; either answer publishable |
| **3** | Fix padding mask; fix or drop all `downstream/*` | 0 | both reviewer-fatal, neither retracts a verdict |
| **4** | Re-analyse: AdamW baseline, 14B parity, α-under-DP | 0 | two headline risks + one free upgrade, all on disk |
| **5** | LR schedule (`lr=lr_for_opt`, one line) | 4–6 runs | XSe alone has stopped converging (§10.5); helps mean *and* variance |
| **6** | Rank **down** to r=8/12 | 6 runs | 201× → 1430–3218× for ≤4e-4, if it holds with seeds |
| **7** | Age-bias correction | 3 runs | law exact to 2 %, but §0 caps the upside |
| **8** | 14B verification with seeds | 6 runs | defensive, only if claiming scale generality |
| — | wavelets, loss-spectral, more α/margin/MP arms | — | don't |
