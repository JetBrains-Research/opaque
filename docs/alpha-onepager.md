# The Rényi order α — one page for the report

Non-DP only. 36 verified runs, no new GPU time. Full detail: `docs/renyi-alpha-theory-final.md`.

---

## What α was supposed to do

LoRA-XSe refreshes some of the 16 frozen directions every few steps. We used a Rényi entropy of
order α to decide **how many to keep**, hoping α would be a useful dial.

α does not act directly. It produces **one whole number per matrix** — how many directions to keep —
and then

> **directions refreshed = 16 − (α's number) − margin**

That whole number is the only way α can influence anything.

---

## The one measurement that everything rests on: "slope"

*Slope* means: **how much does the loss improve if you refresh one more direction?** We measured it
by setting the count by hand:

| directions refreshed | loss | |
|---|---|---|
| 1 | 0.34700 | |
| 5 | 0.34429 | 1 → 5 buys 0.00271, i.e. **0.00068 per direction** (steep) |
| 9 | 0.34368 | |
| 13 | 0.34354 | 9 → 13 buys 0.00014, i.e. **0.000035 per direction** (flat) |

The first few directions matter a lot; past about 9 they barely matter. Twenty-fold difference.

**This decides the α question**, because α's effect can only ever be:

> α's effect = (how far α moves the count) × (what one direction is worth)

If either factor is small, α does nothing. Run-to-run noise is **0.0003** at our operating point — any
effect below that is invisible.

---

## The verdict, in three cases

**α ≥ 1 — does literally nothing.** One direction holds ~90 % of the momentum energy, so every α from
1 to ∞ returns the same number: keep 1. Shannon, α=2 and α=∞ are not three settings, they are **the
same run three times**. Confirmed in 36 of 36 runs.
The differences we chased between them (up to 0.00057) are noise: α moves the count by less than 0.1
of one direction, worth 0.000004 — about 100× too small to explain what we saw.

**α = 0.5 — almost nothing.** Occasionally keeps 2 instead of 1. Moves the count by ~0.7 of a
direction; worth 0.000024. Still ~10× below noise.

**α < 0.5 — real, and always bad.** Below 0.5 it keeps many more directions, so far fewer get
refreshed:

| α | directions refreshed | loss |
|---|---|---|
| ≥ 0.25 | 11 – 13 | 0.34346 |
| 0.20 | 10.3 | 0.34353 |
| 0.15 | 8.9 | 0.34372 |
| 0.10 | 6.5 | 0.34405 |
| 0.05 | 2.8 | **0.34537** |

The last row is 0.0019 worse — about 6× noise, so genuinely real. And it is **monotone: lower α →
shallower refresh → worse result.** There is no sweet spot. α below 0.5 is not a dial to optimise, it
is a cliff to stay away from.

**So α is either inert (≥ 1), nearly inert (0.5), or actively harmful (< 0.5). No setting helps.**

---

## The margin was the real control

Since α's number is always 1, the formula reduces to **refreshed = 15 − margin**. So margin 0 → 15,
1 → 14, 2 → 13. The margin set the depth in every experiment; α set none of it.

**Does that invalidate the α experiments?** No, and we tested it four ways:

1. The α sweep was run at **four margins (0, 1, 2, 3)**, not one.
2. The rule does **not** compensate for a margin change — it **over-reacts**. Raising the margin by 1
   removes 1 direction by construction, and the adaptive part then removes a *further* 0.03–0.57.
   Never gives any back. The over-reaction is worst for the lowest α.
3. The α **ranking does not reproduce** across the four margins — the best α changes each time. A real
   effect would keep the same winner.
4. Most directly: as the margin gives α **3.5× more room** (m=1 → 3), the observed loss spread
   **falls 4.7×** (0.00038 → 0.00008). More authority, less effect — the opposite of a real mechanism.

**Could a much larger margin make α matter?** Yes — around margin 8–10 it would become measurable.
But that region is worse to start with: sitting there costs **0.0008** while α's entire reach there is
**0.0002**. You would give up four times more than you could win back. So no margin makes α worth
tuning.

---

## What to report

1. **α is not a tunable parameter for this method.** For every α ≥ 1 the algorithm is *identical* —
   not similar, identical. Proven, and verified in 36 of 36 runs across four margins.
2. **Keep Shannon (α=1), but present it as a derived constant, not a tuned choice.** The rule is:
   *keep the single strongest momentum direction, refresh the rest.* This is stronger than a tuned α —
   no reviewer can ask us to sweep it.
3. **Report one constraint instead of a tuning range: never go below α ≈ 0.25.** Below that behaviour
   does change, and always for the worse.
4. **Withdraw the "per-matrix adaptive depth" claim.** At α ≥ 1 all 196 matrices received the same
   depth at every step, so there was no per-matrix adaptation to take credit for.
5. **What survives untouched:** rotation itself (the headline result), and the depth rule — *refresh at
   least ~55 % of the directions; above that it makes no difference; going shallow is the only mistake
   and it costs a quarter of the entire benefit of rotation.*
6. **The new theoretical contribution:** an entropy measures **concentration** ("how unevenly is the
   energy spread?"), but the task is **detection** ("which directions are above the noise?"). Different
   questions, different answers — which is why no α could ever have worked. The same statistic is used
   for rank allocation across the parameter-efficient-tuning literature, so the objection travels.

---

## The one thing still genuinely untested

An α contrast at a **shallow** operating point (margin ≥ 4). Only α=1 was ever run there. That is the
only regime where α is identifiable, and it is also the regime where the method is worse — so we
expect nothing useful, but we have not measured it. Cost if wanted: 4 runs.
