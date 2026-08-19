# LoRA-XSe under an adaptive optimizer: what is provable

Non-DP. r=16, r_keep=11, r_e=5. All numerics are CPU linear algebra on synthetic
inputs, used to check algebra -- never to measure the method.

## 0. The operator

At each rotation the r x r core and the optimizer state are re-expressed by a
two-sided map X -> L X Rt, with

    L  = [U_keep^T ; P_B],  P_B = B_explore^T B     (r_e x r)
    Rt = [V_keep  , P_A],   P_A = A A_explore^T     (r x r_e)

Exact structural facts (verified to 1e-15):

* P_B U_keep = 0 and V_keep^T P_A = 0, so L L^T = diag(I_rk, P_B P_B^T). The kept
  rows of L are exactly orthonormal and exactly orthogonal to the explore rows.
* L = diag(I_rk, C) U^T with C = B_explore^T B U_p, so P_B = C U_p^T is supported
  entirely on the DISCARDED columns of U. (The write-up states P_B "has no closed
  form"; it has one.)
* E[||P_B||_F^2] = r_e^2/(d_out - r_keep) EXACTLY = 6.997e-3 at d_out=3584.
* sing(L) = {1 with multiplicity r_keep} UNION sing(C). So L is a contraction,
  never an isometry for r_e > 0, with median cond(L) ~ 234 and E[cond(L)] = +inf.
* rank(L) = r almost surely -- NOT rank r_keep. The gap is conditioning, not rank.
* L is exactly orthogonal iff r_e = 0, or under XSE_RESET_IN_PLACE (where C = I).

So the operator is: an orthogonal change of coordinates into the momentum's
singular basis, then keep r_keep coordinates isometrically and crush the other
r_e by a factor ~0.04. It is a PROJECTION plus injection of fresh directions, not
a rotation.

## 1. The diagnostic we were misreading by an order of magnitude

    ||B' R' A' - B R A||_F / ||B R A||_F  =  sqrt(1 - g^2)   EXACTLY,   g = ||R'||/||R||

and g is precisely the logged rotation/r_norm_growth. The identity is exact (not
asymptotic) because the rotation IS an orthogonal projection of dW, so Pythagoras
applies. Verified to 1e-9 through the real code path.

At the logged median g = 0.978:

    "2.2% of ||R|| lost per rotation"   <- what we had been saying
    4.35% of ENERGY lost
    20.9% RELATIVE CHANGE IN dW         <- what it actually means

The 2.2% is shrinkage along the retained direction; the discarded component is
orthogonal, so it enters the error at first order and the norm only at second.

Corollary: since g <= 1 always, rotation is a strict contraction on ||R||, i.e. an
effective weight decay of 0.74%/step at tau=1 -- roughly 10^3-10^4 times stronger
than the configured lr*wd ~ 1e-6, invisible to the optimizer and absent from every
hyperparameter. That is a live alternative explanation for effects attributed to
"exploration".

Why rotation is nevertheless nearly benign: g = 0.978 implies the kept block holds
95.65% of R's energy while occupying only 47.3% of the slots -- an alignment factor
of 2.02. If R were isotropic, g would be 0.688 and the forward error 72.5%. The
alignment is a property of the trajectory, not of the operator, and bounding it a
priori is the obstacle to any convergence proof.

## 2. Second moment: the target of "exact transport" does not exist

Implemented rule: v_new = (L o L) v (Rt o Rt), clamped, explore band cleared.

* Mass is conserved for all v >= 0 IFF L has unit-norm columns and Rt unit-norm
  rows -- strictly WEAKER than orthogonality, but still failing on every
  coordinate once r_e > 0. Retention is exactly (r_keep/r)^2 = 0.4727 (measured
  0.47285), NOT r_keep/r. Two independent derivations (mass-weighted and
  entry-count) agree.
* The rule is exact iff the gradient history is elementwise uncorrelated. Measured
  p_1 = 0.847 means rank-1 dominance, i.e. MAXIMALLY correlated entries, so the
  assumption is violated about as badly as possible. Error law in that regime: the
  ratio (exact / implemented) is asymptotically a product of two chi-square(1)
  variables -- mean exactly 1, median 0.19, 5th pct 4e-4, 95th 4.8.
* Worst-case bound, no distributional assumption: exact <= r^2 * implemented, so
  the implemented denominator never underestimates by more than a factor r = 16,
  i.e. the step is never more than 16x too large. Attained by a Hadamard witness.
  Overestimation (muting) is unbounded. The error is one-sided in the SAFE
  direction.
* NO-GO. Writing the map in vectorised form K = Rt^T kron L, a diagonal
  preconditioner commutes with it iff the preconditioner is constant on the
  connected components of supp(K). For dense L, Rt there is ONE component, so
  equivariance forces D = c*I -- Adam must degenerate to SGD+momentum.
  Equivariance for arbitrary D holds iff L and Rt are signed permutations.
  **So no rule for computing v_new makes a non-trivial rotation invisible to
  Adam. There is no exact transport to approximate.** Every method in this
  literature -- GaLore's carry, LDAdam's one-sided squared map, PLUMAGE's
  realignment, ours -- is choosing among inexact rules.
* What DOES survive a rotation is only the trace-like aggregate (row sums;
  equivalently the mean predictor, the unique O(r)-equivariant blind estimator by
  Schur's lemma). Per-coordinate structure dies. Independently verified against
  SOAP's Claim 1 and arXiv:2607.05872 Thm 1, which are the same fact from two
  sides.

## 3. The mechanism that predicts a measurement we already had

Rotation contracts v toward its mean by a factor <= sigma_2(L o L):

    r =  4    0.775
    r =  8    0.706
    r = 16    0.578      <- our r
    r = 32    0.447

and sigma_2 is INDEPENDENT of the momentum's rank-1 dominance (swept p_1 from 0.5
to 0.998; sigma_2 did not move), because the singular VECTORS of a spike-plus-noise
matrix stay near-uniformly oriented even when its singular VALUES concentrate.

Steady-state preconditioner condition number (faithful simulation, 30 seeds,
beta2=0.99):

    no rotation          kappa = 11.53
    tau=1,  r_e=0        kappa =  1.12     <- AdamW + pure rotation ~ SGD
    tau=5,  r_e=5        kappa =  3.33
    tau=20, r_e=5        kappa =  2.94

**Rotation destroys ~70% of Adam's adaptive range.** This is a derivation of the
already-measured asymmetry: AdamW buys 6.82e-3 on the frozen arm and 0.16e-3 on
the rotating arm. The earlier "interference" account had no mechanism; this is one.

Constructive consequence: under this cadence the best rotation-blind estimator IS
a scalar, and the implemented map is being driven toward one anyway. A
scalar-preconditioned Adam is therefore the honest optimizer, and it is provably
the gauge-equivariant member of the family.

## 4. Descent direction: the demand cannot be met, and not because of us

* beta1 = 0: the Adam step is a descent direction for ANY v >= 0 whatsoever --
  transport, carry, reset, or garbage. True but VACUOUS as a statement about
  transport; only non-negativity and eps > 0 do any work.
* beta1 > 0: FALSE, and false for unrotated Adam too. Counterexample r=1: g=1,
  m=-1, any v>0 ascends. Heavy-ball and Adam are not per-step descent methods,
  which is why every convergence analysis uses a Lyapunov function containing the
  momentum instead of a descent lemma.
* Positive diagonal preconditioning does not preserve descent even when the
  momentum is a descent direction: g=(1,-1), m=(1,0.5) gives <g,m>=+0.5, but with
  D=diag(1,4), <g,Dm>=-1.0.
* Sharp replacement: with P = sum of positive g_i m_i and N = sum of |negative|,
  <g,Dm> > 0 for every D with condition number kappa IFF P > kappa*N. Combined
  with the fact that the transport does not degrade kappa on the kept block, the
  defensible statement is: "no descent theorem exists for any Adam, transported or
  not; here is the sharp criterion, and our transport does not worsen the half of
  it we control."

LoRA-SB / LoRA-FA / LoRA-Pro all prove descent by a PD-plus-Cholesky argument on
<G, M G N> -- a quadratic form with the SAME vector in both slots. Adam's step is
<g, D m>, different vectors. That is the entire gap, and it does not close.

## 5. "No worse than not rotating" is FALSE

Counterexample: m=n=3, r=2, r_e=1, L(W)=0.5||W-W*||^2 with W* inside the initial
flat. Without rotation the loss reaches exactly 0. With rotation the achievable
minimum inside each new flat is bounded below by 0.5*dist^2(W*, S_new) > 0, and W*
lies in the new flat only on a measure-zero event. Over 20,000 draws the floor has
mean 0.377, median 0.437, and exceeds 0.01 with probability 0.993.

So the defensible claim is not "no worse". It is: for every fixed projection loss
there is a tau above which the exploration gain exceeds it.

## 6. Convergence: do not attempt it for the shipped algorithm

GoLore's Theorem 4 quantifies over ANY subspace optimizer rho: there is an f on
which SVD-based projection fails, for any rank, any tau, and any inner optimizer,
Adam included. No second-moment rule can rescue it under standard assumptions.

A fallback IS provable for beta1 = 0: convergence to a neighbourhood of radius
proportional to Ebar/(eta*tau), where Ebar is the per-rotation projection energy
loss -- a quantity the code already logs. It vanishes iff r_e = 0 or tau -> inf.
For beta1 > 0 (the shipped configuration) this is a CONJECTURE, and closing it is
exactly where LDAdam needs error feedback and an AMSGrad-style clamp that its own
released code does not contain.

Constructive: error feedback on R -- buffer R - L R Rt and re-inject after the
next rotation -- makes Ebar telescope, restores convergence to stationarity, and
is free under the DP post-processing theorem.

## 7. The framing correction that matters most for the paper

Under SGD, PURE rotation (r_e = 0) is a gauge transformation: bit-identical
losses. So every effect measured under SGD comes from r_e > 0, and the paper must
not call it "rotation". Call it what it is: **exploration, sorted by momentum.**

Under Adam the rotation is NOT a gauge transformation (the no-go above), and
section 3 says what it does instead. That turns the objection into an asset.

## 8. Fixed in code as a result of this analysis

Step 5c cleared nu on the explore band while leaving mu transported. That is the
one combination of three that breaks the Cauchy-Schwarz coupling
|mu_hat| <= C sqrt(nu_hat), C = 2.345 at (0.9, 0.99). Faithful simulation
(d_out=3584, 250 trials), worst explore-band step where 1.0 is well-scaled:

    transport both       median 1.001   p95 1.003   max  1.006
    clear nu only (was)  median 1.005   p95 1.062   max 48.995
    clear both (now)     median 1.000   p95 1.000   max  1.000

A tail hazard rather than a systematic one, but a 49x step in 5 of 16 directions
is not worth keeping. Both moments are now cleared together, matching what step 5b
already did under reset-in-place. The xse_sgd path is untouched and verified
bit-identical.

## 9. Open, and cheap

1. p_e = 0 + AdamW + rotation. Section 3 predicts kappa -> 1.12, i.e. it should
   land on SGD-without-rotation. One run, sharp prediction.
2. XSE_ADAM_STATE = {transport, carry, reset}. The switch exists. Turns the
   transport question into a measurement.
3. Scalar-preconditioned Adam as a fourth arm.
4. Error feedback on R.
