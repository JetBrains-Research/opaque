# LoRA-XS and rotation: the mathematics, from the ground up

Written to be read start to finish. Every number is measured on this project's own
runs (Qwen2.5-Coder-7B, $r=16$, $r_{keep}=11$, $r_e=5$, $d_{out}=d_{in}=3584$,
196 adapter sites) unless marked as an identity.

---

## 1. The problem: a weight matrix is enormous

One weight matrix in a 7B model is $d_{out} \times d_{in} = 3584 \times 3584 \approx 12.8$
million numbers. Fine-tuning means finding a change $\Delta W$. Done directly, that
is 12.8M trainable numbers per matrix, across 196 matrices.

## 2. LoRA: assume the change is low-rank

LoRA writes

$$\Delta W = B \cdot A, \qquad B \in \mathbb{R}^{d_{out} \times r},\quad A \in \mathbb{R}^{r \times d_{in}},\quad r = 16$$

The illuminating way to read this is **as a sum of outer products**:

$$\Delta W = \sum_{i=1}^{r} b_i\, a_i^{\top}$$

Each term $b_i a_i^{\top}$ is one **wire**: *read the input along direction $a_i$, write
the output along direction $b_i$.* So LoRA gives you **16 wires and lets you choose
both endpoints of each**, at a cost of $16 \times (3584 + 3584) = 114{,}688$ numbers.

## 3. LoRA-XS: freeze the endpoints, train the switchboard

LoRA-XS fixes the directions in advance -- from the SVD of the pretrained $W_0$ --
and inserts a small square matrix between them:

$$\Delta W = B \cdot R \cdot A, \qquad R \in \mathbb{R}^{r \times r}$$

$B$'s columns are $W_0$'s top-$r$ left singular vectors, $A$'s rows its top-$r$ right
singular vectors. **Both frozen.** Only $R$ trains: $16 \times 16 = 256$ numbers per
site, **50,176** in total.

Expand again:

$$\Delta W = \sum_{i=1}^{r}\sum_{j=1}^{r} R_{ij}\; b_i\, a_j^{\top}$$

**$R$ is a full bipartite coupling matrix.** Entry $R_{ij}$ is the strength of the wire
from input direction $a_j$ to output direction $b_i$.

That is the real trade, and it is not what is usually said:

| | wires | endpoints | trainable |
|---|---|---|---|
| LoRA $r=16$ | 16 | **free** | 114,688 |
| LoRA-XS $r=16$ | **256** (all pairings) | **fixed** | 256 |

LoRA-XS is **richer inside its subspace** ($r^2$ pairings rather than $r$) but
**confined** to that subspace.

## 4. The cage, quantified

The reachable set is

$$\Delta W \in \operatorname{col}(B) \otimes \operatorname{row}(A)$$

a **fixed 256-dimensional linear subspace** of the 12.8-million-dimensional space of
matrices -- about 0.002% of the available directions, chosen once, before training
starts.

The measurement that makes this interesting:

| basis | share of the gradient's energy captured |
|---|---|
| $W_0$'s top-16 singular directions | **0.077%** |
| a *random* 256-dimensional subspace | **0.022%** |

The "informed" choice is only **3.4x better than random**, and the advantage does not
grow with rank. For `v_proj` it is *below* random in 3 of 5 sampled layers. The cage
was never good; it was cheap.

## 5. Rotation: move the cage

If the cage is the problem, move it. But *where*? The signal is the **momentum of $R$**
-- the accumulated gradient, i.e. where $R$ has consistently wanted to go with
per-step noise averaged out.

Take its SVD:

$$m = U S V^{\top}$$

**The singular vectors live in the 16-dimensional coordinate space of $R$, not in the
big space.** So $u_i$ is a *recipe for mixing your current 16 output directions*, and
$B u_i$ is a new output direction -- still inside $\operatorname{col}(B)$, just re-mixed.

The rotation is then:

1. **Keep** the top $r_{keep}=11$ mixtures: $B_{kept} = B\,U_{keep}$. Inside the old span.
2. **Draw** $r_e=5$ fresh directions $B_{explore}$ **outside** the old span.
3. **New basis** $B_{new} = [\,B_{kept} \mid B_{explore}\,]$, and symmetrically for $A$.

Each rotation re-mixes what it keeps and swaps 5 of 16 dimensions for genuinely new
ones. **That is how the cage moves.**

## 6. Re-expressing $R$: the two-sided map

$R$ was written in the old basis. What should $R_{new}$ be? Whatever changes the model
least -- the **orthogonal projection** onto the new subspace:

$$\pi(\Delta W) = B_{new}\big(B_{new}^{\top}\, \Delta W\, A_{new}^{\top}\big)A_{new}$$

Reading off the middle factor:

$$\boxed{\;R_{new} = \underbrace{(B_{new}^{\top} B)}_{L}\; R\; \underbrace{(A A_{new}^{\top})}_{Rt} = L\,R\,Rt\;}$$

$L$ says how each **new** output direction decomposes in the **old** ones; $Rt$ does the
same on the input side. Using $B^{\top}B = I$:

$$L = \begin{bmatrix} U_{keep}^{\top} \\ P_B \end{bmatrix}, \qquad P_B = B_{explore}^{\top} B$$

## 7. The uncomfortable truth: it is not a rotation

$P_B$ measures how much the *fresh* directions overlap the *old* span. A random
direction in 3584 dimensions barely touches a 16-dimensional subspace:

$$\mathbb{E}\,\|P_B\|_F^2 = \frac{r_e^{2}}{d_{out}-r_{keep}} = 6.997\times 10^{-3}, \qquad \|P_B\|_F \approx 0.083$$

(This is exact, not asymptotic. $P_B$ annihilates $\operatorname{col}(U_{keep})$
exactly, and is supported entirely on the discarded columns of $U$.) Therefore

$$L \approx \begin{bmatrix} U_{keep}^{\top} \\ \approx 0 \end{bmatrix}$$

**$L$ is a projection with 5 nearly-empty rows, not a rotation.** Its singular values
are eleven exact $1$s and five of size $\approx 0.04$. It is still full rank almost
surely -- median condition number $\approx 234$ -- so the issue is *conditioning*, not
rank.

So "rotation" actually means: **project onto the top-11 momentum subspace, then bolt on
5 empty slots.** The exception is `XSE_RESET_IN_PLACE`, where the fresh directions are
drawn *inside* the old span, giving $C = I$ and $L = U^{\top}$ exactly orthogonal.

**The exact cost of a rotation.** With $B,A$ orthonormal,

$$\frac{\|B'R'A' - BRA\|_F}{\|BRA\|_F} = \sqrt{1-g^{2}}, \qquad g = \frac{\|R'\|_F}{\|R\|_F}$$

This is an identity, not an approximation, because projection makes Pythagoras exact.
*Proof.* $\|BRA\|_F^2 = \|R\|_F^2$ by orthonormality; the cross term is
$\langle B'R'A', BRA\rangle = \operatorname{tr}(R'^{\top} L R Rt) = \|R'\|_F^2$; expand
the square. $\blacksquare$

And $g$ is exactly the logged `rotation/r_norm_growth`. At the measured median
$g = 0.978$:

| reading | value |
|---|---|
| "2.2% of $\|R\|$ lost per rotation" | what we had been saying |
| energy lost, $1-g^2$ | 4.35% |
| **relative change in $\Delta W$, $\sqrt{1-g^2}$** | **20.9%** |

Every rotation moves the adapter delta by a fifth of its own norm. The 2.2% is only
the shrinkage *along the retained direction*; the discarded part is orthogonal, so it
enters the error at first order and the norm only at second.

**Why it is nevertheless nearly benign.** $g = 0.978$ implies the kept block holds
95.65% of $R$'s energy while occupying only 47.3% of the slots -- an alignment factor
of $2.02$. If $R$ were isotropic, $g$ would be $0.688$ and the forward error 72.5%.
$R$ has concentrated itself into the momentum's dominant subspace. That is a property
of the *trajectory*, not of the operator, and bounding it a priori is the obstacle to
any convergence proof.

**A corollary nobody had noticed.** Since $g \le 1$ always, rotation is a strict
contraction on $\|R\|$: a multiplicative leak of $0.74\%$ per step at $\tau=1$. A
configured weight decay contributes $\eta \cdot wd \approx 10^{-6}$. **Rotation acts as
a decay term $10^3$--$10^4$ times stronger than any weight decay we set**, invisible to
the optimizer and absent from every hyperparameter.

---

# 8. Momentum translates exactly. Here is why.

Momentum is an accumulated gradient, $m = \sum_s w_s g_s$. The gradient of $R$ lives in
the *same* coordinate space as $R$, so under a basis change it transforms identically:

$$g_{new} = B_{new}^{\top} (\nabla_W L) A_{new}^{\top} = L\, g\, Rt$$

Momentum is a **linear** combination of gradients and $L(\cdot)Rt$ is a **linear** map,
so they commute:

$$\boxed{\;m_{new} = L\, m\, Rt\;}$$

**Exact, with no assumptions.** And there is a beautiful special case. Since
$m = U S V^{\top}$, while $L$'s kept rows are $U_{keep}^{\top}$ and $Rt$'s kept columns
are $V_{keep}$:

$$(L\,m\,Rt)_{\text{keep,keep}} = U_{keep}^{\top}\,(U S V^{\top})\,V_{keep} = \operatorname{diag}(S_{1..r_{keep}})$$

**The rotation diagonalises the momentum by construction.** Immediately after a
rotation all momentum sits on the diagonal -- measured off-diagonal magnitude
$2.6\times10^{-20}$ -- which is why the code simply writes $\operatorname{diag}(S)$
there.

One caveat on the word "exact": $L\,m\,Rt$ is exactly the new-basis momentum of the
history *projected onto the old span*. It equals the momentum the new basis "would have
had" only when $r_e = 0$. The gap is the history's gradient energy along directions
that were never observed -- which is the whole point of exploration.

# 9. The second moment does not translate. Here is why.

Adam keeps a second buffer, $v = \mathrm{EMA}[g^{\circ 2}]$ -- **elementwise** square,
the typical *size* of each coordinate's gradient. Its purpose is per-coordinate step
sizing, $\text{step} = m/\sqrt{v}$.

The obstruction in one sentence: **squaring and mixing do not commute.**

Write the transform entrywise:

$$(L g Rt)_{ij} = \sum_{k,l} L_{ik}\, g_{kl}\, Rt_{lj}$$

Square it and take expectations:

$$v'_{ij} = \sum_{k,l}\sum_{k',l'} L_{ik}L_{ik'}\,Rt_{lj}Rt_{l'j}\; \mathbb{E}\!\left[g_{kl}\,g_{k'l'}\right]$$

Split into the terms where $(k,l) = (k',l')$ and the rest:

$$v'_{ij} = \underbrace{\big(L^{\circ 2}\, v\, Rt^{\circ 2}\big)_{ij}}_{\text{the implemented map}} \;+\; \underbrace{\sum_{(k,l)\neq(k',l')} L_{ik}L_{ik'}Rt_{lj}Rt_{l'j}\,\mathbb{E}[g_{kl}g_{k'l'}]}_{\text{everything Adam does not store}}$$

The squared map is exactly the **diagonal part**. The remainder needs cross-terms
between *different* entries.

## The intuition, in two coordinates

Two coordinates, each of variance 1. Mix them: $(x+y)/\sqrt{2}$.

- **Independent:** variance $= 1$.
- **Perfectly correlated** ($x = y$): variance $= 2$.

Same diagonal $(1,1)$; different answer after mixing.

> **A vector rotates because it has a direction. A variance has no direction -- it is a
> size -- and the size of a mixture depends on correlations that a diagonal never
> recorded.**

## The information-theoretic wall

Exactness needs the full covariance: $r^2(r^2+1)/2 = 32{,}896$ numbers. Adam stores its
diagonal: $256$. Two different covariances with the same diagonal produce different
$v'$.

> **No function of $v$ alone can be an exact transport.** That is not a defect of our
> formula; it is a limit of what Adam carries.

## And the biggest missing piece is the mean, not the correlations

$$\mathbb{E}[g_{kl}g_{k'l'}] = \underbrace{\operatorname{Cov}(g_{kl},g_{k'l'})}_{\text{noise}} \;+\; \underbrace{M_{kl}M_{k'l'}}_{\text{mean},\; M=\mathbb{E}[g]}$$

The dropped terms are **raw second moments, not covariances**. So even with *perfectly
uncorrelated* entries the mean products survive, and they vanish for every pair only if
$M$ has at most one nonzero entry. The true exactness condition is therefore
**orthogonality in $L^2$** -- uncorrelated *and* essentially mean-free -- which is far
stronger than independence.

Measured: the error is essentially **rank-independent** (relative Frobenius $0.886$ at
rank 1, $0.871$ at rank 16). **It is not the momentum's rank-1 structure that breaks the
formula; it is that the gradient has a mean at all.** Rank-1 dominance matters only
because it implies the mean is large.

Worst case, with the new basis aligned to $g$'s singular vectors -- exactly what the
rotation arranges -- the dominant entry of $v$ comes out $p_1 r^2 = 213\times$ too
small at $p_1=0.85$, $r=16$, inflating that step $14.6\times$.

## The fix, which needs no new state

$m$ already estimates $M$, and $m$ already transports linearly. So split the mean off
exactly and give the squared map only the fluctuation:

$$\boxed{\;v_{new} = k_t\,(L\,m\,Rt)^{\circ 2} \;+\; (L^{\circ 2})\,\big(v - k_t\,m^{\circ 2}\big)_{+}\,(Rt^{\circ 2})\;}$$

with $k_t = (1-\beta_2^{t})/(1-\beta_1^{t})^{2}$ reconciling the two zero-init biases
(for a constant $g$, $m = (1-\beta_1^t)g$ and $v = (1-\beta_2^t)g^{\circ 2}$, hence
$v = k_t\,m^{\circ 2}$ exactly). No Jensen factor belongs here: the momentum's own
variance in the signal term exactly cancels the deficit in the residual term.

**Signal transported linearly, then squared. Noise given the mass-transport map.**
Exact for any deterministic gradient at any rank; reduces identically to the old rule
as $m \to 0$, so nothing regresses in the noise-dominated or DP limit. This is LDAdam's
rule (arXiv:2410.16103), generalised to the two-sided case; the old rule was PLUMAGE's
(arXiv:2505.18313).

# 10. Two structural facts worth carrying in your head

**(a) The squared map is a mass transport.** When $L$ is orthogonal, $L^{\circ 2}$ is
*doubly stochastic* -- every row and column sums to 1 -- so it redistributes $v$'s total
without creating or destroying it. Hence mass is conserved exactly when $r_e = 0$, and
retention is exactly $(r_{keep}/r)^2 = 0.4727$ when $r_e = 5$ (each side independently
keeps $r_{keep}/r$; the map is two-sided, so the fraction squares).

The sharp criterion is weaker than orthogonality: mass is conserved for every
$v \ge 0$ **iff** $L$ has unit-norm columns and $Rt$ unit-norm rows.

**(b) Doubly stochastic means averaging, and averaging destroys adaptivity.** A doubly
stochastic map can only flatten a vector, never concentrate it. The contraction rate
toward the mean is

$$\sigma_2\!\left(L^{\circ 2}\right) = 0.578 \quad \text{at } r=16$$

so **each rotation removes about 42% of $v$'s deviation from its mean** -- and $\sigma_2$
is *independent* of the momentum's rank-1 dominance (swept $p_1$ from 0.5 to 0.998; it
does not move). With $\tau=1$ and $\beta_2=0.99$, $v$ spans roughly 100 rotations, so it
is flattened almost to a constant. Measured steady-state condition number of the
preconditioner:

| configuration | $\kappa$ |
|---|---|
| no rotation | **11.53** |
| $\tau=1$, $r_e=0$ | 1.12 |
| $\tau=5$, $r_e=5$ | 3.33 |

> **Adam's entire value is that $v$ differs across coordinates. Rotation averages $v$
> across coordinates. They work against each other.**

That single fact *derives* a measurement we already had: AdamW improves the frozen arm
by $6.82\times10^{-3}$ and the rotating arm by $0.16\times10^{-3}$.

# 11. The no-go: a better rule cannot fix it

Suppose you had the exact $v_{new}$. Would rotation then be invisible to Adam? **No.**

In the small-$\varepsilon$ limit Adam's step is $\approx \operatorname{sign}(g)$.
Invisibility would require

$$\operatorname{sign}(L\,g\,Rt) = L\,\operatorname{sign}(g)\,Rt$$

and $\operatorname{sign}$ is not linear.

Formally, vectorise the map as $K = Rt^{\top} \otimes L$. A diagonal preconditioner $D$
satisfies $D_{new} K = K D_{old}$ for all inputs **iff** $D$ is constant on the
connected components of the bipartite support graph of $K$. For dense $L, Rt$ there is
**one** component, forcing $D = c\,I$ -- i.e. **Adam has degenerated into SGD with
momentum**. Equivariance for arbitrary $D$ holds **iff** $L$ and $Rt$ are *signed
permutations*, i.e. iff the "rotation" merely relabels and flips coordinates.

> **The target of "exact second-moment transport" does not exist.** GaLore's carry,
> LDAdam's rule, PLUMAGE's, ours -- all are inexact rules with nothing exact to
> approximate. What survives a rotation is only the trace-like aggregate (the row sums,
> equivalently the unique $O(r)$-equivariant blind estimator by Schur's lemma); the
> per-coordinate detail is provably lost.

# 12. Everything on one page

| object | how it changes under a rotation | difficulty |
|---|---|---|
| basis $B, A$ | keep 11 momentum mixtures, draw 5 fresh | the mechanism itself |
| core $R$ | $L R\,Rt$ -- the optimal projection | exact map, but loses $\sqrt{1-g^2} = 21\%$ of $\Delta W$ |
| momentum $m$ | $L\,m\,Rt$ -- the same map | **exact**, and it diagonalises $m$ |
| second moment $v$ | mass transport $L^{\circ 2} v\, Rt^{\circ 2}$ | **provably cannot be exact**; needs the mean split |

**The one-sentence version.** LoRA-XS trades free endpoints for a dense switchboard
inside a fixed 256-dimensional cage; rotation walks the cage by keeping the momentum's
dominant mixtures and injecting fresh directions; the core and the momentum follow that
move exactly because both are linear objects; and the second moment cannot follow it at
all, because a variance has no direction to be rotated.

## A naming consequence worth acting on

Under SGD, **pure** rotation ($r_e = 0$) is a *gauge transformation*: it produces
bit-identical losses, because $L$ and $Rt$ are orthogonal and SGD-with-momentum is
rotation-equivariant. So every effect measured under SGD comes from $r_e > 0$, not from
rotating. The honest name for the mechanism is **exploration, sorted by momentum**.

Under Adam the rotation is *not* a gauge transformation -- section 11 -- and section 10
says what it does instead.
