# Phase 1 — math note: Mellum2's training objective as a per-example DP objective

Agent: `math`. Scripts and raw output:
`/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/math/validate_identities.py`
and `.../math/validate_identities.out` (float64, CPU, tiny shapes; every identity below
is checked there — reported max-abs-diffs are quoted inline).

Evidence tags: **VERIFIED** = I read the cited lines / ran the cited script;
**PLAUSIBLE** = derived or read secondary text but not executed end-to-end;
**REFUTED** = shown false. Repo citations are `file:line` relative to
`/home/user/opaque`; the HF file is the venv copy
`.venv/lib/python3.11/site-packages/transformers/models/mellum/modeling_mellum.py`
(transformers 5.16.1), abbreviated `modeling_mellum.py`.

---

## 0. Notation and the two facts everything rests on

Batch $\mathcal B$ of examples $x$; token positions $t=1..T$ (right-padded); attention mask
$m_{x,t}\in\{0,1\}$; $T_x=\sum_t m_{x,t}$; $T_{\rm tot}=\sum_{x\in\mathcal B}T_x$.
Layers $l=1..L$; router logits $z^l_{x,t}\in\mathbb R^E$; full-softmax probabilities
$p^l_{x,t}=\operatorname{softmax}(z^l_{x,t})$ (over **all** $E$ experts, fp32 —
`modeling_mellum.py:335`); top-$k$ index set $S^l_{x,t}=\operatorname{topk}_k(z^l_{x,t})$;
indicator $I^l_{x,t,e}=\mathbf 1\{e\in S^l_{x,t}\}$.

**Fact A (the batch-coupled object).** `load_balancing_loss_func`
(`modeling_mellum.py:540-606`, **VERIFIED**) keeps *one* `tokens_per_expert_sum`
(`:575`), *one* `router_prob_sum` (`:576`) and *one* `total_rows` (`:577`) across all
layers, accumulating per layer (`:596-600`) and normalising once at the end
(`:602-603`):

$$
f_e(\mathcal B)=\frac{1}{L\,T_{\rm tot}}\sum_{l}\sum_{x}\sum_{t} m_{x,t}\,I^l_{x,t,e},\qquad
P_e(\mathcal B)=\frac{1}{L\,T_{\rm tot}}\sum_{l}\sum_{x}\sum_{t} m_{x,t}\,p^l_{x,t,e},
$$
$$
L_{\rm aux}(\mathcal B)=E\sum_{e=1}^{E} f_e(\mathcal B)\,P_e(\mathcal B)\qquad(\text{`:605-606`}).
$$

Consequences: $\sum_e f_e(\mathcal B)=k$ (each token contributes $k$ assignments;
verified: 2.0 for $k=2$), $\sum_e P_e(\mathcal B)=1$, so at perfect balance
$f\equiv k/E$ the loss equals $k$ (Mellum2: $8$, i.e. $\alpha L_{\rm aux}=0.008$ nats at
$\alpha=10^{-3}$). The mask is the **attention** mask (`:694`, passed at `:697`), not the
label mask: prompt tokens with label $-100$ still count in the aux loss. This is Switch
Transformer eqs. (4)–(6), §2.2 (https://arxiv.org/abs/2101.03961, **VERIFIED** via
arXiv HTML: $\text{loss}=\alpha N\sum_i f_iP_i$, $f_i=\frac1T\sum_{x}\mathbf 1\{\arg\max p(x)=i\}$,
$P_i=\frac1T\sum_x p_i(x)$, "only the $P$-vector is differentiable"), generalised by HF
to top-$k$ (counts sum to $k$, not 1) and pooled over layers.

**Fact B (per-example decomposition of the batch statistics).** Define per-example,
token-mean, layer-pooled quantities
$$
P_e(x)=\frac{1}{L\,T_x}\sum_l\sum_t m_{x,t}\,p^l_{x,t,e},\qquad
f_e(x)=\frac{1}{L\,T_x}\sum_l\sum_t m_{x,t}\,I^l_{x,t,e},\qquad
w_x=\frac{T_x}{T_{\rm tot}}.
$$
Then exactly $P_e(\mathcal B)=\sum_x w_x P_e(x)$ and $f_e(\mathcal B)=\sum_x w_x f_e(x)$
(verified: $|L_{\rm aux}^{\rm HF}-E\sum_e f_e(\mathcal B)P_e(\mathcal B)|\le 9\times10^{-16}$).
So the *only* batch coupling is the bilinear product of two token-weighted means.

---

## 1. Gradient of the batch aux loss and the per-example surrogate

### 1.1 Gradient

$I^l_{x,t,e}$ is a function of the ordering of $z^l_{x,t}$, hence locally constant in
$\theta$ off the measure-zero tie set $\{\exists\,e\ne e': z_e=z_{e'}\}$. Therefore
$\nabla_\theta f_e(\mathcal B)=0$ a.e. (torch: `topk` returns indices with no gradient,
`bincount`/`scatter_add_` of the mask are gradient-free; `modeling_mellum.py:582-600`).
By the product rule,
$$
\boxed{\;\nabla_\theta L_{\rm aux}(\mathcal B)=E\sum_e f_e(\mathcal B)\,\nabla_\theta P_e(\mathcal B)
=\frac{E}{L\,T_{\rm tot}}\sum_{x}\sum_{t}\sum_{l}m_{x,t}\sum_e f_e(\mathcal B)\,\nabla_\theta p^l_{x,t,e}\;}
$$
(The task's "$\frac{E}{T_{\rm tot}}\sum_x\sum_t\sum_e$" is the single-layer case; with HF's
layer pooling the prefactor is $E/(L\,T_{\rm tot})$ and there is a sum over $l$.)
At the token level, with $c=f(\mathcal B)$ held fixed,
$$
\frac{\partial}{\partial z_j}\sum_e c_e\,p_e = p_j\Big(c_j-\sum_e c_e p_e\Big)
\qquad(\text{verified: max diff }0.0).
$$

### 1.2 The separable surrogate

For any fixed vector $\tilde f\in\mathbb R^E$ define
$$
S(x;\tilde f)=E\,w_x\sum_e \tilde f_e\,P_e(x)=\frac{E\,T_x}{T_{\rm tot}}\sum_e\tilde f_e P_e(x).
$$
Then $\sum_x S(x;f(\mathcal B))=L_{\rm aux}(\mathcal B)$ identically (Fact B), and because
$\nabla f(\mathcal B)=0$ a.e.,
$$
\boxed{\;\nabla_\theta L_{\rm aux}(\mathcal B)=\sum_x\nabla_\theta S(x;\tilde f)\Big|_{\tilde f=f(\mathcal B)}\;}
$$
**VERIFIED**: `max|grad aux_HF − grad Σ_x S(x;f(B))|` $=1.0\times10^{-17}$ (full mask) and
$2.3\times10^{-17}$ (ragged masks), against $\|\nabla L_{\rm aux}\|\approx 0.05$–$0.08$.

So: *treat the load vector as a constant and the batch aux loss is exactly a sum of
per-example losses.* The whole DP problem reduces to obtaining a legitimate constant
$\tilde f$ (Sections 3–5).

### 1.3 Normalisation: which per-example loss reproduces the HF batch gradient

HF's total training loss is (`modeling_mellum.py:689-700`; `loss_utils.py:33-48`,
`ForCausalLMLoss` with `num_items_in_batch`, **VERIFIED**)
$$
\mathcal L^{\rm HF}(\mathcal B)=\frac{1}{T^{\rm lab}_{\rm tot}}\sum_x\sum_t m^{\rm lab}_{x,t}\,\mathrm{CE}_{x,t}
+\alpha L_{\rm aux}(\mathcal B)
=\sum_x w^{\rm lab}_x\,\mathrm{CE}_x+\alpha\sum_x S(x;f(\mathcal B)),
$$
where $\mathrm{CE}_x$ is the per-example token-mean CE over label-valid tokens,
$w^{\rm lab}_x=T^{\rm lab}_x/T^{\rm lab}_{\rm tot}$ (label mask, $-100$), and $w_x$ uses the
attention mask. A DP pipeline computes $\frac1B\sum_x\nabla\ell_x$. Matching HF exactly
requires
$$
\ell^{\rm HF\text{-}match}_x(\theta;\tilde f)=B\Big[w^{\rm lab}_x\,\mathrm{CE}_x(\theta)+\alpha\,E\,w_x\sum_e\tilde f_eP_e(x;\theta)\Big],
$$
which contains the batch statistics $T^{\rm lab}_{\rm tot},T_{\rm tot}$ and is therefore
**not** an admissible per-example loss (its per-example sensitivity would depend on the
other examples). Opaque's convention (`packages/opaque-alignment/src/opaque/api/alignment/sft/loss/_nll.py:15-24, 86-87`,
**VERIFIED**: "pre-clip division by the per-example token count") is equal example
weights. The consistent aux term under that convention is
$$
\boxed{\;\ell_x(\theta;\tilde f)=\mathrm{CE}_x(\theta)+\alpha\,E\sum_e\tilde f_e\,P_e(x;\theta)\;}
$$
and the batch-level objective it implements is $\frac1B\sum_x[\mathrm{CE}_x+\alpha E\sum_e\tilde f_eP_e(x)]$
— the *example-mean* instead of HF's *token-mean*. The two coincide exactly iff
$T_x\equiv T$ (and $T^{\rm lab}_x\equiv T^{\rm lab}$), in which case $w_x=1/B$ and also
$f(\mathcal B)=\frac1B\sum_x f(x)$. **VERIFIED**: full masks → equal-weight
$\frac1B\sum_x\nabla\ell_x$ matches HF to $8\times10^{-18}$; ragged masks → mismatch
$7.3\times10^{-3}$ (vs. gradient norm $5.0\times10^{-2}$), and the reweighted
$B\,w_x\,\ell_x$ restores agreement to $2.3\times10^{-17}$, and $|f(\mathcal B)-\bar f|$ up to
$0.042$ where $\bar f=\frac1B\sum_x f(x)$. For the Mellum2 presets (packed seq 1024;
BRIEF "Existing DP presets") the mismatch is nil; for ragged FIM data it is the same
example-vs-token reweighting already accepted for CE and documented in `_nll.py`. The
released estimate $\tilde f$ should use the *same* weighting as the loss: with equal
example weights release $\bar f=\frac1B\sum_x f(x)$ (Section 3a), with token weights
release counts (Section 3b).

Two HF-side caveats that define "what $f(\mathcal B)$ actually is" in the reference:
(i) $f(\mathcal B)$ is the **micro-batch** load vector, computed per forward call
(`:693`); (ii) under gradient accumulation `trainer.py:1961-1963` (**VERIFIED** read)
divides by the accumulation count *only* when the model does not accept loss kwargs.
Mellum accepts `num_items_in_batch`, so the CE part is globally token-normalised while
the $\alpha L_{\rm aux}$ part is summed across micro-batches undivided — the effective aux
coefficient in HF is $\alpha\times$(accumulation steps) (**PLAUSIBLE**: read, not run).
The surrogate with $\tilde f$ = a whole-(logical-)batch estimate is thus *closer* to the
intended objective than HF's own accumulation behaviour; "faithful to HF" should mean
"faithful to Fact A on the logical batch".

---

## 2. Bias of a per-example aux loss

Let $\ell^{\rm own}_x=E\sum_e f_e(x)P_e(x)$ (the "compute $f$ on the example's own
tokens" variant) and $\ell^{\rm sur}_x=E\sum_e\tilde f_eP_e(x)$. With $\nabla f(x)=0$ a.e.,
$$
\boxed{\;\nabla\ell^{\rm own}_x-\nabla\ell^{\rm sur}_x=E\sum_e\big(f_e(x)-\tilde f_e\big)\nabla P_e(x)\;}
$$
**VERIFIED** per example: identity holds to $\le 9.7\times10^{-17}$; the batch-mean of the
own-variant differs from the HF gradient by $77\%$ (full) / $219\%$ (ragged) relative
$L_2$ on random routers. Token-level form with $d=f(x)-\tilde f$:
$\partial_{z_j}=\frac{E}{L T_x}\,p_j\big(d_j-\sum_e d_ep_e\big)$.

**When is it large — the specialisation model.** Suppose documents fall into types
(languages, file kinds) and a document of type $\tau$ routes (nearly) all its tokens into a
type-specific set $S_\tau$, $|S_\tau|=k$, with the batch balanced across types so that
$f(\mathcal B)=k/E\cdot\mathbf 1$. Then:

* Batch aux gradient: $E\sum_e\frac kE\nabla P_e(x)=k\,\nabla\!\sum_eP_e(x)=k\,\nabla 1=0$.
  The regulariser is *silent* at balance, as designed. **VERIFIED**: $\|\nabla L_{\rm aux}\|=1.5\times10^{-16}$.
* Own-aux gradient: with $f(x)=\mathbf 1_{S_x}$ the token-level bracket is
  $d_j-\sum_ed_ep_e=\mathbf 1\{j\in S_x\}-P_{S_x}$, so
  $$
  \nabla_{z}\ell^{\rm own}_x\big|_{t}=\frac{E}{L T_x}\nabla_z P_{S_x}(t),\qquad P_{S}(t)=\sum_{e\in S}p_{t,e},
  $$
  i.e. $\ell^{\rm own}_x=E\cdot\overline{P_{S_x}}$: it pushes every document's probability
  mass *out of the experts it actually uses*, with per-token logit gradient
  $\frac{E}{LT_x}p_j(1-P_S)$ on own experts and $-\frac{E}{LT_x}p_jP_S$ on the others,
  regardless of batch balance. **VERIFIED** (toy $E=8,k=2$, 4 types, 8 docs): own-aux value
  $7.94\approx E$ vs batch aux $2=k$; $\|\nabla\|=0.335$ vs $1.5\times10^{-16}$; closed form
  matches to $0.0$.

For Mellum2 ($E=64$, $\alpha=10^{-3}$) the own-aux term has value up to $\alpha E=0.064$ nats
and a *systematic*, never-vanishing anti-specialisation gradient of relative strength
$\alpha E$ per unit of $P_S$; the batch/surrogate term is $\alpha k=0.008$ nats and vanishes
at balance. Small in absolute terms, but it is a different regulariser (it penalises
within-document routing concentration, which is exactly what a code MoE wants), so
$\ell^{\rm own}$ is **not** a faithful stand-in. This settles H2's second half.

Value bias: $\ell^{\rm own}_x-\ell^{\rm sur}_x=E\sum_e(f_e(x)-\tilde f_e)P_e(x)$ — the
within-document covariance of hard counts and soft probabilities, typically positive.

---

## 3. The load-histogram release: definitions and sensitivity

Per-example histogram (per layer, per expert):
$h^{(L)}_x\in\mathbb R^{L\times E}$, $h^{(L)}_{x,l,e}=\frac1{T_x}\sum_tm_{x,t}I^l_{x,t,e}$;
pooled $h_x\in\mathbb R^E$, $h_{x,e}=\frac1L\sum_lh^{(L)}_{x,l,e}=f_e(x)$.
Since HF pools over layers (Fact A), **the surrogate only needs the pooled $E$-vector**;
the per-layer vector is diagnostic.

Constraints: $0\le h^{(L)}_{x,l,e}\le1$, $\sum_eh^{(L)}_{x,l,e}=k$ for every $l$; pooled:
$0\le h_{x,e}\le1$, $\sum_eh_{x,e}=k$.

**Norm bounds.** For any $l$: $\sum_e(h_{l,e})^2\le\max_eh_{l,e}\sum_eh_{l,e}\le k$, hence
$$
\|h^{(L)}_x\|_2\le\sqrt{kL},\quad\|h^{(L)}_x\|_1=kL;\qquad
\|h_x\|_2\le\sqrt k,\quad\|h_x\|_1=k,
$$
attained when every token in every layer routes to the same $k$ experts; the balanced
lower bound is $\|h_x\|_2\ge k/\sqrt E$. Centering (public constant, post-processing
inverse): $\|h_x-\frac kE\mathbf 1\|_2^2=\|h_x\|^2-k^2/E\le k(1-k/E)$. **VERIFIED** numerically
(adversarial $\|h^{(L)}\|_2=\sqrt{kL}$, pooled $=\sqrt k$, random routing well below).

**Sensitivity under add/remove adjacency** (the repo default,
`.junie/differential-privacy-review.md` "Adjacency"): the query $q(\mathcal B)=\sum_xh_x$
changes by one term, so $\Delta_2=\sup_x\|h_x\|_2$, $\Delta_1=\sup_x\|h_x\|_1$:

| release | $\Delta_2$ | $\Delta_1$ | needs clipping? |
|---|---|---|---|
| (a) pooled fractions, equal weights | $\sqrt k$ ($\sqrt{k(1-k/E)}$ centred) | $k$ | no — bound is structural |
| (a') per-layer fractions | $\sqrt{kL}$ | $kL$ | no |
| (b) token-weighted counts $c_x=T_xh_x$ | $T_{\max}\sqrt k$ | $T_{\max}k$ | no if $T_x\le T_{\max}$ is a public collator constant (it is: seq 1024); otherwise clip |

Under replace-one adjacency double these (review protocol, "twice the sensitivity").
The divisor must be a public constant (expected batch size $\bar B$ = opaque's
`normalize_by`), never the realised Poisson batch size or $T_{\rm tot}$.

**Gaussian mechanism error.** Release $\hat f=\frac1{\bar B}\big(\sum_xh_x+\mathcal N(0,\sigma_h^2\Delta_2^2I)\big)$.
Per-entry noise std $\sigma_h\Delta_2/\bar B$; against the signal $f_e\approx k/E$:
$$
r_{\rm pooled}=\frac{\sigma_h\sqrt k\,E}{\bar B\,k}=\frac{\sigma_hE}{\bar B\sqrt k},\qquad
r_{\rm per\text{-}layer}=\frac{\sigma_hE\sqrt L}{\bar B\sqrt k},\qquad
\sigma_h(r)=\frac{r\,\bar B\sqrt k}{E}\ \text{(pooled)}.
$$
Token-weighting (b) has identical $r$ when $T_x\equiv T_{\max}$ and is worse otherwise (the
signal shrinks, the bound does not). Under Poisson sampling this release is subsampled by
the *same* coin as the gradient — see 4(c).

---

## 4. Accounting options for the extra release

Throughout: gradient group with per-example bound $C_g$, histogram group with bound
$C_h$ (either the structural $\lambda\sqrt k$ for a scaled copy $\lambda h_x$, or a clip),
noise multiplier $\sigma$ ("nm"), public divisor $\bar B$.

### 4(a) Joint clipping: one vector, one bound

$v_x=[g_x;\lambda h_x]$, clip $\|v_x\|_2\le C$, release $\sum_xv_x+\mathcal N(0,\sigma^2C^2I)$.
Accounting: exactly `gaussian(σ)`; **zero** extra $\varepsilon$. Costs:

* *Gradient signal.* $\|g_x\|^2+\lambda^2\|h_x\|^2\le C^2$ with $\lambda^2\|h_x\|^2\le\lambda^2k=:\rho^2C^2$:
  the gradient's admissible norm shrinks to $C\sqrt{1-\rho^2}$ (when $h$ saturates; less for
  balanced examples), and clipping rescales $g_x$ and $h_x$ by the *same* factor, so the
  released histogram is a biased (down-scaled) count for clipped examples.
* *Load noise.* $\hat f=\frac{1}{\lambda\bar B}\sum_x[\text{clipped }\lambda h_x]+\mathcal N(0,\frac{\sigma^2C^2}{\lambda^2\bar B^2})$,
  per-entry std $\frac{\sigma C}{\lambda\bar B}$; with $\lambda=\rho C/\sqrt k$:
  $r=\frac{\sigma E}{\rho\,\bar B\sqrt k}$.

### 4(b) Separate groups: which noise is correct for a Gaussian on the concatenation

Mechanism $M(D)=q(D)+\mathcal N(0,\Sigma)$ with $\Sigma=\operatorname{diag}(\sigma_g^2I_{d_g},\sigma_h^2I_{d_h})$
and $q$ the concatenated per-group-clipped sum. Whitening $\Sigma^{-1/2}$ is a bijection,
so $M$ and $\Sigma^{-1/2}M=\tilde q+\mathcal N(0,I)$ are mutual post-processings and have the
same privacy. The $L_2$ sensitivity of $\tilde q$ under add/remove is
$$
\Delta(\tilde q)=\sup_x\Big\|\Sigma^{-1/2}\begin{bmatrix}g_x\\ h_x\end{bmatrix}\Big\|_2
=\sup_x\sqrt{\frac{\|g_x\|^2}{\sigma_g^2}+\frac{\|h_x\|^2}{\sigma_h^2}}
=\sqrt{\frac{C_g^2}{\sigma_g^2}+\frac{C_h^2}{\sigma_h^2}}=:\frac1{\sigma_{\rm eff}},
$$
with equality attained by an example that saturates both bounds. The mechanism is
therefore a sensitivity-1 Gaussian at noise multiplier $\sigma_{\rm eff}$; its tight
dominating pair is $(\mathcal N(0,1),\mathcal N(1/\sigma_{\rm eff},1))$
(Zhu–Dong–Wang, https://arxiv.org/abs/2106.08567, Def. 7 and the Gaussian pair
$P=\mathcal N(0,\sigma^2),Q=\mathcal N(\Delta,\sigma^2)$ — **VERIFIED** via arXiv HTML), i.e.
PLD $=$ `gaussian(σ_eff)`. Hence the **Mahalanobis constraint**
$$
\boxed{\;\frac{C_g^2}{\sigma_g^2}+\frac{C_h^2}{\sigma_h^2}=\frac1{\sigma^2}\iff\text{accounting }=\texttt{gaussian}(\sigma)\;}
$$
Three admissible allocations and one inadmissible one:

| allocation | $\sigma_g$ | $\sigma_h$ | constraint value | verdict |
|---|---|---|---|---|
| isotropic | $\sigma\sqrt{C_g^2+C_h^2}$ | same | $1/\sigma^2$ | correct (docs `clipping.md:397`) |
| MSE-optimal, equal dims (opaque) | $\sigma\sqrt{C_g(C_g+C_h)}$ | $\sigma\sqrt{C_h(C_g+C_h)}$ | $1/\sigma^2$ | correct |
| budget split $\eta$ | $\sigma C_g/\sqrt{1-\eta}$ | $\sigma C_h/\sqrt\eta$ | $1/\sigma^2$ | correct (Andrew et al. Thm 1 form) |
| naive per-group $\sigma C_g$, $\sigma C_h$ | $\sigma C_g$ | $\sigma C_h$ | $K/\sigma^2$ | **REFUTED** as `gaussian(σ)`: it is `gaussian(σ/√K)` |

MSE-optimal derivation: minimise $\sum_gd_g\sigma_g^2$ s.t. $\sum_gC_g^2/\sigma_g^2=1/\sigma^2$;
Lagrange gives $\sigma_g^2=C_g\sqrt{\mu/d_g}$; for equal $d_g$, $\sigma_g=\sigma\sqrt{C_g\sum_hC_h}$.

**What opaque implements (VERIFIED by reading and by running):**
`packages/opaque-engine/src/opaque/api/engine/noise_allocation.py:44-110` —
`per_group_noise_stddev` returns `noise_multiplier * sqrt(c * sum_c)` (`:107`) with the
docstring stating the constraint $\sum_i(C_i/n)^2/\sigma_i^2\le1/\mathrm{nm}^2$ (`:57-58`)
and the "equal group dimensions" caveat (`:55-56`); `gaussian_noise` applies it when
`max_norm` is `PerGroup` (`packages/opaque-dpsgd/src/opaque/api/dpsgd/noise/_gaussian.py:320-322`);
`ClippedPytree.noise_stddev_for` offers `optimal`/`isotropic`
(`packages/opaque-engine/src/opaque/api/engine/types.py:280-338`); tests pin the
constraint (`packages/opaque-dpsgd/tests/noise/test_per_group_noise_stddev.py:31-45`)
and the formula (`packages/opaque-engine/tests/types/test_clipped_pytree_noise_stddev.py:61-70`).
My script calls the real function: $\sum C_g^2/\sigma_g^2=0.34602=1/\mathrm{nm}^2$ (opaque) vs
$0.69204=2/\mathrm{nm}^2$ (naive). **Matches the theory.**

Two practical remarks. (i) The equal-dimension MSE objective is the wrong objective
here (the histogram has 64 entries against $10^6$–$10^9$ gradient entries, and we care about
*relative* error of $f$, not total MSE). The right knob is the budget split $\eta$; under
opaque's optimal allocation $\eta=C_h/(C_g+C_h)=\rho/(1+\rho)$ with $\rho=C_h/C_g$, under
isotropic $\eta=\rho^2/(1+\rho^2)$ — so $\lambda$ (the scale of $h$ inside the group) is how
you choose $\eta$ without touching the allocation code. (ii) With the group bound
$C_h=\lambda\sqrt k$ being structural, no clipping happens on the histogram group, so the
release is unbiased (unlike 4(a)).

Costs under the optimal allocation with $\rho=C_h/C_g$:
gradient noise $\times\sqrt{1+\rho}$; histogram per-entry std
$\frac{\sigma\sqrt{k(1+\rho)/\rho}}{\bar B}$; $r=\frac{\sigma E}{\bar B\sqrt k}\sqrt{\tfrac{1+\rho}{\rho}}$.
Precedent: Andrew et al. 2021 (https://arxiv.org/abs/1905.03871, §2.1 and Theorem 1;
**VERIFIED** via arXiv HTML) release the clipped-count bit with its own $\sigma_b$ and set
$z_\Delta=(z^{-2}-(2\sigma_b)^{-2})^{-1/2}$ — exactly the $\eta$ split with
$C_h=\tfrac12$, $\sigma_h=\sigma_b$.

### 4(c) Separate mechanism composed via PLD

If the histogram is released by an *independent* Gaussian mechanism `gaussian(σ_h)`
(sensitivity $\sqrt k$) each step and the gradient by `gaussian(σ)`, adaptive composition
(Zhu–Dong–Wang Thm 10, **VERIFIED**) gives per-step PLD `gaussian(σ) ⊗ gaussian(σ_h)`.
Two Gaussians with independent noise on the *same* batch are, however, precisely the
diagonal-covariance mechanism of 4(b), so the exact per-step PLD is
$$
\texttt{gaussian}(\sigma_{\rm eff}),\qquad \sigma_{\rm eff}^{-2}=\sigma^{-2}+\sigma_h^{-2},
$$
(in GDP terms $\mu_{\rm eff}=\sqrt{\mu^2+\mu_h^2}$). **Subsampling caveat**: with Poisson
sampling the gradient and histogram releases share the sampling coin. The adaptive
composition theorem composes mechanisms with *fresh* internal randomness given previous
outputs; the second release's subsampling is not fresh, so "compose two subsampled
Gaussians" does not describe the mechanism that runs. The joint-vector view does: one
subsampled Gaussian at $\sigma_{\rm eff}$ (equivalently, 4(b) with the $\eta$ split). Use it.
Where there is no amplification (full batch, or DP-FTRL with b-min-sep), 4(c) and 4(b)
coincide exactly.

$\varepsilon$ overhead formula: $\varepsilon_{\rm extra}=\varepsilon_{\rm PLD}(\sigma_{\rm eff},q,T)-\varepsilon_{\rm PLD}(\sigma,q,T)$
with $\sigma_{\rm eff}=(1+\sigma^2/\sigma_h^2)^{-1/2}\sigma$; numeric $\varepsilon$ left to the
primitives agent. At $\sigma=1$: $\sigma_h=1\to\sigma_{\rm eff}=0.707$; $2\to0.894$; $5\to0.981$;
$10\to0.995$.

### 4(d) Lagged / EMA use is free

Let $y_{<t}$ be all noisy outputs before step $t$ and $\tilde f_t=\phi(y_{<t})$ any
function (last release, EMA $\tilde f_t=\beta\tilde f_{t-1}+(1-\beta)\hat f_{t-1}$, clamp to
$[0,1]$, renormalise to sum $k$…). Step $t$'s mechanism is
$M_t(D;y_{<t})=\sum_{x\in\mathcal B_t}\operatorname{clip}_C\nabla\ell_x(\theta_t;\tilde f_t)+\mathcal N(0,\sigma^2C^2)$.
For **every** $y_{<t}$ the per-example contribution is bounded by $C$ (clipping does not
care what $\tilde f_t$ is), so the same dominating pair dominates $M_t(\cdot;y_{<t})$ for all
prefixes, and Thm 10 (adaptive composition of dominating pairs; also Abadi et al. 2016's
moments accountant and Dwork–Rothblum–Vadhan's $k$-fold adaptive composition, which
DP-SGD already relies on because $\theta_t$ itself is a function of $y_{<t}$) yields the
product PLD. **The dependence of the loss on previous noisy outputs is the same kind of
dependence as on $\theta_t$; it costs nothing.** The only paid item is the histogram
release itself (4a/4b/4c). EMA statistics: at stationarity the noise variance of
$\tilde f$ is $\frac{1-\beta}{1+\beta}$ times a single release's, at the price of a lag
$\approx1/(1-\beta)$ steps (bias only while $f$ drifts).

Same-step variant: a forward-only vmap pass computes $h_x$, release $\hat f_t$, then the
gradient pass uses $\tilde f_t=\phi(\hat f_t,y_{<t})$ — two mechanisms per step, still
adaptive composition, still one extra release; costs a routing-only forward. The lagged
variant is what fits `DPTrainer._augment_inputs`
(`packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py:2302-2305`,
"runs once per step … outside vmap", **VERIFIED**): inject $\tilde f_t$ as a broadcast
(non-batched) input tensor; it is public, so it may also be `all_reduce`d freely under DDP.
Serialize $\tilde f_t$/EMA state with the checkpoint (public post-processing; needed for
reproducibility, not privacy).

---

## 5. DP-FTRL (matrix mechanism)

Denisov et al. 2022 (https://arxiv.org/abs/2202.08312, Theorem 2.1 and Def. 3.1;
**VERIFIED** via arXiv HTML): for $A=BC$ and neighbouring streams $G,H$ (differ in one row,
$\|{\rm row\ diff}\|_2\le\zeta$) with $\|C(G-H)\|_F\le\kappa$, $\mathcal M(G)=B(CG+Z)$,
$Z\sim\mathcal N(0,\kappa^2\sigma^2)$, satisfies the same DP guarantee "even when the rows of
the input are chosen adaptively". Requirements on any extra release: constant per-step
row bound $\zeta$ (the strategy is optimised against a fixed $\Delta$ —
`docs/user-guide/noise.md:225-228`, latch `_validate_constant_max_norm` in
`packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_mf_gaussian_noise.py:163`).

**Option (i) — same stream, extra group (recommended).** Row $t$ is
$[\,\sum_x\operatorname{clip}_{C_g}g_x\;;\;\lambda\sum_xh_x\,]$; per-group per-row bound
$(C_g,\lambda\sqrt k)$ is constant by construction. Opaque's dispatcher already accepts
`PerGroup` max_norm and applies `per_group_noise_stddev(max_norm, nm) · row_l2(t)`
(`_mf_gaussian_noise.py:166-167, 182-188`, **VERIFIED** read); accounting is
`MfGaussian(nm, strategy).pld() = gaussian(nm / strategy.sensitivity(...))`
(`packages/opaque-dpftrl/src/opaque/api/accounting/dpftrl/mechanisms/_mf_gaussian.py:1-8, 118-`).
Correctness of the per-group MF: whitening per group gives
$\sum_g\|C(G_g-H_g)\|_F^2/\sigma_g^2\le\operatorname{sens}(C)^2\sum_gC_g^2/(\mathrm{nm}^2C_gS)=\operatorname{sens}(C)^2/\mathrm{nm}^2$,
because the participation-pattern sensitivity is homogeneous of degree 1 in the row
bound — same PLD as the scalar case. Bonus: the MF strategy is optimised so that
*prefix sums* $A\hat G$ are accurate; the running average of the histogram stream is
therefore accurate "for free" — precisely the smoothed $\tilde f$ we want, with the
correlated noise doing the EMA's job.

**Option (ii) — separate fresh Gaussian per step, composed.** Valid in the
non-amplified setting as far as the per-step mechanisms go, but the MF mechanism is one
mechanism whose outputs are released *interleaved* with the $T$ histogram Gaussians whose
outputs feed the adaptive rows. Covering that interleaving rigorously needs concurrent
composition (Vadhan–Wang 2021 / Vadhan–Zhang 2023 — theorem numbers **not verified**
here; **PLAUSIBLE**). Option (i) needs only Denisov Thm 2.1. Cleaner: (i).

**Varying $\tilde f_t$.** $\tilde f_t=\phi(y_{<t})$ is post-processing of previous
mechanism outputs; Thm 2.1's adaptive-rows clause is exactly this situation; the row's
clipped norm bound is unchanged by $\tilde f_t$. No violation. The one thing that *would*
violate MF is a data-dependent, drifting bound (adaptive clipping) — not introduced here.
b-min-sep / balls-in-bins participation: the histogram group shares the gradient's
participation pattern (same example, same step), so the same `min_sep`,
`max_participations` apply.

---

## 6. Routing pinning

Per-example loss with recomputed routing: $\ell_x(\theta)=F(\theta;r_x(\theta))$ where
$r_x(\theta)=\{S^l_{x,t}(\theta)\}$ is piecewise constant in $\theta$ and $F(\theta;r)$ is
smooth for fixed $r$ (softmax, top-$k$ renormalisation over a *fixed* index set
(`modeling_mellum.py:337-338`), SwiGLU experts, attention). torch's `topk` passes gradient
only through the selected values, so within a piece the autograd gradient is
$\nabla_\theta F(\theta;r)$ — this is already what runs (BRIEF: "gradient stopped only
through argmax"). Across a piece boundary (a near-tie) $r$ changes and
$\theta\mapsto\nabla\ell_x(\theta)$ jumps by $O(1)$.

**Pinned routing:** $r_x$ computed by a fixed per-example function $\pi(x)$ (frozen base
model, or the current model evaluated once per step outside vmap in fp32 and passed as a
constant input). Then $\ell_x(\theta)=F(\theta;\pi(x))$ is $C^\infty$ in $\theta$ on all of
parameter space (the pieces are fixed per example): the per-example gradient is
continuous, and for two numerically perturbed evaluations
$\|\nabla\ell_x(\theta)-\nabla\ell_x(\theta')\|\le\mathrm{Lip}\cdot\|\theta-\theta'\|$ instead of
an $O(1)$ jump. **VERIFIED** (toy with an exact tie): perturbing $\theta$ by $\pm10^{-9}$
changes the recomputed-top-$k$ gradient by $1.27$ and the pinned gradient by
$4\times10^{-17}$. This is the mechanism behind H3 (bf16 round-off in vmap vs. eager
flipping routes and producing the observed DP-vs-oracle drift).

**Sensitivity / DP:** both variants are functions of $(x,\theta_t,\text{public model})$
only. The clipping bound $\|\operatorname{clip}\nabla\ell_x\|\le C$ is all the theorem uses;
smoothness is irrelevant to privacy. Pinning from the current model computed *on the
private example* is still a per-example function. No DP issue either way — provided
the frozen base model is public (it is: a released checkpoint) and the pinned routes are
never released (they are consumed inside the per-example computation only). The
histogram $h_x$ then uses the pinned routes, consistently.

**Utility trade-off:** frozen-base pinning freezes the *training-time* routing while the
inference router (whose inputs move through attention LoRA even if the router weights are
frozen) keeps drifting → train/inference mismatch that grows with the size of the
fine-tune; current-model pinning (recomputed each step, fp32, outside vmap, fed as
constants) has no mismatch beyond bf16-vs-fp32 tie-breaking and removes the
vmap-vs-eager flip source of drift, at the cost of one routing-only forward per step
(router linear + softmax + top-$k$ on every layer's input — but the inputs depend on the
previous layers, so it is a full forward without expert gradients). Middle ground: pin
from the current model but under vmap in fp32 with a deterministic tie rule; that removes
precision flips but not the discontinuity across steps (harmless for DP; harmless for
utility since routes are supposed to move).

---

## 7. Worked numbers for Mellum2 ($L=28,E=64,k=8,T=1024,\bar B=256,\sigma=1$)

Histogram sensitivities (add/remove; double for replace-one):

| quantity | value |
|---|---|
| pooled $\Delta_2=\sqrt k$ | 2.828 (centred $\sqrt{k(1-k/E)}=2.646$) |
| pooled $\Delta_1=k$ | 8 |
| per-layer $\Delta_2=\sqrt{kL}$ | 14.97 |
| per-layer $\Delta_1=kL$ | 224 |
| token-count pooled $\Delta_2=T\sqrt k$ | 2896 |
| aux value at balance $=k$; $\alpha\cdot k$ | 8; 0.008 nats |

Separate release `gaussian(σ_h)` on the pooled vector (signal $f_e=k/E=0.125$):

| $\sigma_h$ | per-entry std $\sigma_h\sqrt k/\bar B$ | relative error $r$ (single step) | $r$ after EMA $\beta=0.9$ / $0.95$ / $0.99$ | $\sigma_{\rm eff}$ at $\sigma=1$ |
|---|---|---|---|---|
| 1 | 0.0110 | 8.8 % | 2.0 % / 1.4 % / 0.6 % | 0.707 |
| 2 | 0.0221 | 17.7 % | 4.1 % / 2.8 % / 1.3 % | 0.894 |
| 5 | 0.0552 | 44.2 % | 10.1 % / 7.1 % / 3.1 % | 0.981 |

(per-layer histogram: multiply $r$ by $\sqrt L=5.29$ → 46.8 % at $\sigma_h=1$; not needed.)
Required $\sigma_h$ for single-step $r=10\%$: $1.13$; for $r=10\%$ *after* EMA($\beta=0.95$): $7.1$
(then $\sigma_{\rm eff}=0.990$, i.e. a $\sim1\%$ change in the noise multiplier).

Joint options at the same $\sigma=1$ (fraction $\rho=C_h/C_g$ of the gradient bound given
to $\lambda h_x$, $\lambda=\rho C_g/\sqrt k$), single-step $r$ before EMA:

| $\rho$ | 4(a) joint clip: $r$, gradient budget $\sqrt{1-\rho^2}$ | 4(b) optimal: $r$, gradient noise $\times\sqrt{1+\rho}$ | 4(b) isotropic: $r$, gradient noise $\times\sqrt{1+\rho^2}$ |
|---|---|---|---|
| 0.1 | 88 %, 0.995 | 29 %, 1.049 | 89 %, 1.005 |
| 0.3 | 29 %, 0.954 | 18 %, 1.140 | 31 %, 1.044 |
| 0.5 | 18 %, 0.866 | 15 %, 1.225 | 20 %, 1.118 |

Reading: with EMA $\beta\ge0.95$ every option lands at a few-percent relative error on
$f$; the cheapest in gradient terms is the $\eta$/budget-split form of 4(b) with a small
$\eta$ (e.g. $\sigma_h=5$–$7$: $\le2\%$ gradient-noise inflation, $\le10\%$ $r$ after EMA),
which in opaque is realised either as `PerGroup` clipping with a suitably small $\lambda$ or
as an explicit `gaussian(σ_eff)` composed accountant. $\varepsilon$ numbers: primitives agent,
using $\sigma_{\rm eff}=(1+\sigma^2/\sigma_h^2)^{-1/2}\sigma$.

---

## 8. Summary of the per-example objective (what to implement)

Per step $t$, public constant $\tilde f_t\in\mathbb R^E$ ($\sum_e\tilde f_{t,e}=k$ after
renormalisation, entries clamped to $[0,1]$), per-example loss
$$
\ell_x(\theta;\tilde f_t)=\mathrm{CE}_x(\theta)+\alpha\,E\sum_{e}\tilde f_{t,e}\,P_e(x;\theta),\qquad
P_e(x;\theta)=\frac1{LT_x}\sum_{l,t}m_{x,t}\,\operatorname{softmax}(z^l_{x,t}(\theta))_e,
$$
whose batch mean has *exactly* HF's aux gradient when $\tilde f_t=f(\mathcal B_t)$ and
$T_x\equiv T$ (Section 1), and the released per-example histogram
$h_x=f(x)\in[0,1]^E$, $\|h_x\|_2\le\sqrt k$ (Section 3), fed either as a second `PerGroup`
group through the same Gaussian / MF mechanism (Sections 4b, 5(i)) or as a separate
`gaussian(σ_h)` whose exact joint PLD is `gaussian(σ_eff)` (Section 4c), consumed
lagged/EMA at no further cost (Section 4d). Pin routes per example (Section 6) so the
per-example gradient is continuous in $\theta$.

## 9. Open items

* $\varepsilon$ overhead numbers for `gaussian(σ_eff)` under Poisson $q=B/N$ and band-MF —
  primitives agent.
* Concurrent-composition theorem numbers (Vadhan–Wang 2021; Vadhan–Zhang 2023) if
  option 5(ii) is ever preferred — not verified here.
* HF gradient-accumulation scaling of the aux term (`trainer.py:1961-1963`) was read,
  not executed; verify with a 2-microbatch run if "faithful to HF Trainer" (as opposed
  to "faithful to Fact A") is the target.
* Whether Mellum2 fine-tuning data are actually unbalanced enough for $\alpha=10^{-3}$ to
  matter with router/experts trainable (H4) — empirical, out of scope for this note.
