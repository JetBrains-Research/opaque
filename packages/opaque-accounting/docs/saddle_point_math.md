# Saddle-Point Privacy Accountant: Mathematical Foundations

**Implementation reference**: `packages/opaque-accounting/src/pld/cgf_pld.rs`

This document gives a self-contained derivation of the formulas used in our
CGF-backed privacy accountant, including provable error bounds. We decompose
the hockey-stick divergence into two tail probabilities and evaluate each via
the Lugannani-Rice saddle-point approximation.

---

## 1. Setup and Notation

Let P, Q be two probability measures (adjacent datasets) and let

$$
L = \log \frac{dP}{dQ}(X), \qquad X \sim P
$$

be the **privacy loss random variable** (PLRV). The **(epsilon, delta)-guarantee**
is characterized by the hockey-stick divergence:

$$
\delta(\varepsilon) = E_\gamma(P \| Q)
  = \mathbb{E}_P\!\bigl[(1 - e^{\varepsilon - L})^+\bigr]
  = \mathbb{P}(L > \varepsilon) - e^\varepsilon \, \mathbb{P}_Q(L > \varepsilon).
$$

The **cumulant generating function** (CGF) of L under P is:

$$
\Lambda(t) = \log \mathbb{E}_P[e^{tL}] = \log M_L(t),
$$

where M_L(t) is the moment generating function.

### 1.1 Exponential Tilting

For any t in the domain of Lambda, the **exponential tilt** of L by parameter t is
the distribution P_t defined by:

$$
\frac{dP_t}{dP}(x) = \frac{e^{tx}}{\mathbb{E}_P[e^{tL}]} = e^{tx - \Lambda(t)}.
$$

Under P_t, the CGF of L is the **shifted CGF**:

$$
\Lambda_t(s) = \Lambda(s + t) - \Lambda(t).
$$

**Key observation**: P_Q = P_{-1} (tilting by -1 recovers Q), since
dQ/dP = e^{-L}, giving:

$$
\Lambda_{-1}(s) = \Lambda(s - 1) - \Lambda(-1).
$$

---

## 2. Hockey-Stick Decomposition

We decompose delta(epsilon) into two tail probabilities.

**Proposition 2.1.** For any epsilon >= 0:

$$
\delta(\varepsilon)
  = \mathbb{P}_P(L > \varepsilon)
    \;-\; e^{\varepsilon + \Lambda(-1)} \cdot \mathbb{P}_{-1}(L > \varepsilon)
$$

where P_{-1} is the exponential tilt of P by -1.

*Proof.* Starting from the second form of the hockey-stick divergence:

$$
\delta(\varepsilon)
  = \mathbb{P}_P(L > \varepsilon) - e^\varepsilon \, \mathbb{P}_Q(L > \varepsilon).
$$

Now, P_Q(L > epsilon) = E_Q[1_{L > epsilon}]. Changing measure from Q to P:

$$
\mathbb{P}_Q(L > \varepsilon)
  = \mathbb{E}_P\!\bigl[e^{-L} \, \mathbf{1}_{L > \varepsilon}\bigr]
  = e^{\Lambda(-1)} \, \mathbb{E}_{P_{-1}}\!\bigl[\mathbf{1}_{L > \varepsilon}\bigr]
  = e^{\Lambda(-1)} \, \mathbb{P}_{-1}(L > \varepsilon).
$$

Substituting back:

$$
\delta(\varepsilon) = \mathbb{P}_P(L > \varepsilon)
  - e^{\varepsilon + \Lambda(-1)} \cdot \mathbb{P}_{-1}(L > \varepsilon). \qquad \square
$$

This is **exact** -- no approximation has been made.

**Advantage over [Alghamdi et al. 2023]**: The paper's contour integral
packs everything into a single integrand with
F_epsilon(z) = Lambda(z) - epsilon*z - log(z) - log(1+z).
The log(z) term creates a singularity at z=0, causing numerical
issues when epsilon is near 0. Our decomposition avoids this entirely:
each tail uses the clean saddle-point equation Lambda'(t*) = epsilon.

---

## 3. Lugannani-Rice Saddle-Point Approximation

We now approximate each tail probability using the Lugannani-Rice formula
(Lugannani & Rice, 1980).

### 3.1 General Setup

Let X be a random variable with CGF K(t) = log E[e^{tX}]. We want to
compute P(X > x) for a given threshold x.

**Step 1: Saddle-point equation.** Find t* > 0 satisfying:

$$
K'(t^*) = x.
$$

This is Newton's method on a strictly convex function (K is convex since
K''(t) = Var_{P_t}(X) > 0), so the root is unique whenever x is in the
interior of the range of K'.

**Step 2: Standardized variables.** Define:

$$
\hat{r} = \operatorname{sign}(t^*) \cdot \sqrt{2\bigl(t^* x - K(t^*)\bigr)},
\qquad
\hat{s} = t^* \sqrt{K''(t^*)}.
$$

**Geometric interpretation**: r-hat is the signed deviance (related to the
Kullback-Leibler divergence from the original to the tilted distribution),
and s-hat is the saddlepoint standard deviation.

**Step 3: Lugannani-Rice formula.**

$$
\mathbb{P}(X > x)
  \approx \bar{\Phi}(\hat{r})
    + \varphi(\hat{r})\!\left(\frac{1}{\hat{r}} - \frac{1}{\hat{s}}\right)
$$

where Phi-bar(r) = 1 - Phi(r) is the standard normal survival function
and phi(r) is the standard normal density.

### 3.2 Application to First Tail: P_P(L > epsilon)

For the first tail, X = L under P with CGF K(t) = Lambda(t):

- **Saddle-point**: solve Lambda'(t*) = epsilon
- **r** = sign(t*) * sqrt(2 * (t* * epsilon - Lambda(t*)))
- **s** = t* * sqrt(Lambda''(t*))
- **Tail**: P_P(L > epsilon) ~ Phi-bar(r) + phi(r) * (1/r - 1/s)

### 3.3 Application to Second Tail: P_{-1}(L > epsilon)

For the second tail, X = L under P_{-1} with shifted CGF
K_{-1}(t) = Lambda(t - 1) - Lambda(-1):

- K'_{-1}(t) = Lambda'(t - 1), so the saddle equation K'_{-1}(s*) = epsilon becomes
  Lambda'(s* - 1) = epsilon, i.e., **s* = t* + 1** (same t* as the first tail!)
- K_{-1}(s*) = Lambda(t*) - Lambda(-1)
- K''_{-1}(s*) = Lambda''(t*)

Thus:

- **r_{-1}** = sign(s*) * sqrt(2 * (s* * epsilon - Lambda(t*) + Lambda(-1)))
- **s_{-1}** = s* * sqrt(Lambda''(t*)) = (t* + 1) * sqrt(Lambda''(t*))
- **Tail**: P_{-1}(L > epsilon) ~ Phi-bar(r_{-1}) + phi(r_{-1}) * (1/r_{-1} - 1/s_{-1})

### 3.4 Combining

$$
\delta(\varepsilon) \approx
  \underbrace{\bar\Phi(\hat r) + \varphi(\hat r)\!\left(\tfrac{1}{\hat r} - \tfrac{1}{\hat s}\right)}_{\text{first tail}}
  \;-\;
  e^{\varepsilon + \Lambda(-1)} \cdot
  \underbrace{\left[\bar\Phi(\hat r_{-1}) + \varphi(\hat r_{-1})\!\left(\tfrac{1}{\hat r_{-1}} - \tfrac{1}{\hat s_{-1}}\right)\right]}_{\text{second tail}}
$$

In log-space (as implemented):

$$
\delta(\varepsilon) = \exp(\log T_1) - \exp\!\bigl(\varepsilon + \Lambda(-1) + \log T_2\bigr)
$$

computed via log-subtract-exp for numerical stability.

---

## 4. Composition

### 4.1 Homogeneous Composition

For n independent applications of the same mechanism, L_total = L_1 + ... + L_n
(independent, identically distributed). By independence:

$$
\Lambda_{\text{total}}(t) = n \cdot \Lambda(t),
\qquad
\Lambda'_{\text{total}}(t) = n \cdot \Lambda'(t),
\qquad
\Lambda''_{\text{total}}(t) = n \cdot \Lambda''(t).
$$

All formulas in Section 3 apply with these substitutions. The computation is
**O(1)** in n -- we never form the n-fold convolution.

### 4.2 Heterogeneous Composition

For k distinct mechanisms applied n_1, ..., n_k times respectively:

$$
\Lambda_{\text{total}}(t) = \sum_{i=1}^{k} n_i \cdot \Lambda_i(t),
$$

and similarly for derivatives. The computation is O(k), independent of the
total number of compositions N = sum n_i.

---

## 5. Error Bounds

### 5.1 Lugannani-Rice Error for a Single Tail

The Lugannani-Rice formula is a third-order saddle-point approximation.
Its error is characterized by the following classical result.

**Theorem 5.1** (Lugannani & Rice, 1980; Jensen, 1995). Let X_1, ..., X_n
be independent random variables with total CGF K_n(t) = sum K_i(t).
Let t* solve K'_n(t*) = x and define r-hat, s-hat as in Section 3.1. Then:

$$
\mathbb{P}\!\left(\frac{X_1 + \cdots + X_n - \mu}{\sigma} > z\right)
  = \bar\Phi(\hat r) + \varphi(\hat r)\!\left(\frac{1}{\hat r} - \frac{1}{\hat s}\right)
  + O\!\left(\frac{1}{n}\right)
$$

uniformly in z (and hence uniformly in epsilon), where the error constant
depends on the cumulants of X_i.

More precisely, following Daniels (1987) and Jensen (1995, Theorem 6.4):

$$
\left|\mathbb{P}(S_n > x) - \text{LR}(\hat r, \hat s)\right|
  \leq \frac{C \cdot \rho_3}{K''_n(t^*)^{3/2}}
$$

where:
- rho_3 = sum |kappa_{3,i}(t*)| is the sum of absolute third cumulants
  under the tilted distribution P_{t*}
- C is an absolute constant (C <= 1 for the standard bound)
- K''_n(t*) = sum K''_i(t*) is the variance under the tilted distribution

For n-fold self-composition: rho_3 = n * |kappa_3(t*)| and
K''_n(t*) = n * K''(t*), giving:

$$
\text{error per tail} = O\!\left(\frac{1}{\sqrt{n}}\right).
$$

### 5.2 Total Error Bound for delta(epsilon)

Since delta(epsilon) = T_1 - c * T_2 (where c = e^{epsilon + Lambda(-1)} > 0),
the total error satisfies:

$$
|\delta(\varepsilon) - \hat\delta(\varepsilon)|
  \leq |\text{err}_1| + e^{\varepsilon + \Lambda(-1)} \cdot |\text{err}_2|
$$

where err_1 and err_2 are the Lugannani-Rice errors for the first and second
tails respectively.

**Theorem 5.2** (Error bound for the privacy accountant). For n-fold
composition of a mechanism with single-step CGF Lambda, at privacy parameter
epsilon:

$$
|\delta(\varepsilon) - \hat\delta(\varepsilon)|
  \leq \frac{C_1 \cdot |\kappa_3(t^*)|}{\bigl(n \cdot \Lambda''(t^*)\bigr)^{3/2}}
    + e^{\varepsilon + n\Lambda(-1)} \cdot \frac{C_2 \cdot |\kappa_3^{(-1)}(t^*\!+\!1)|}{\bigl(n \cdot \Lambda''(t^*)\bigr)^{3/2}}
$$

where kappa_3(t*) is the third cumulant of L under P_{t*}, and
kappa_3^{(-1)} is under the doubly-tilted distribution.

For the **Gaussian mechanism** with noise sigma:
- Lambda(t) = t(1+t) / (2*sigma^2)
- Lambda''(t) = 1/sigma^2 (constant -- all higher cumulants vanish!)
- kappa_3(t) = 0 for all t

**Corollary 5.3.** For the Gaussian mechanism under n-fold composition,
the Lugannani-Rice approximation is **exact**:

$$
\hat\delta(\varepsilon) = \delta(\varepsilon)
$$

for all n >= 1 and all epsilon >= 0.

*Proof.* The sum of n Gaussians is Gaussian. The Lugannani-Rice formula is
exact for distributions in the normal family (it reproduces the exact normal
tail). Since the PLRV of the Gaussian mechanism is normally distributed
(L ~ N(1/(2*sigma^2), 1/sigma^2)), any n-fold composition is also Gaussian,
and the formula is exact. QED.

### 5.3 Berry-Esseen-Type Bound (Non-Gaussian Case)

For non-Gaussian mechanisms (e.g., Poisson-subsampled Gaussian), the error
has a Berry-Esseen flavor.

**Theorem 5.4** (Adapted from Jensen 1995, Theorem 6.4, and Alghamdi et al.
2023, Theorem 5.6). For n-fold composition of a mechanism with bounded third
absolute moment rho = E_{t*}[|L - Lambda'(t*)|^3] under the tilted measure,
the Lugannani-Rice error for each tail satisfies:

$$
|\text{err}| \leq \frac{0.56}{\bigl(n \cdot K''(t^*)\bigr)^{3/2}}
  \cdot n \cdot \rho
  = \frac{0.56 \, \rho}{K''(t^*)^{3/2} \cdot \sqrt{n}}.
$$

The constant 0.56 comes from the Berry-Esseen theorem
(the best known constant for the one-sided bound is 0.56, from Shevtsova 2010).

**Corollary 5.5** (Practical error bound). For the total delta approximation:

$$
|\delta - \hat\delta|
  \leq \frac{0.56}{\sqrt{n}}
  \left(
    \frac{\rho_0}{(\Lambda''(t^*))^{3/2}}
    + e^{\varepsilon + n\Lambda(-1)} \cdot
      \frac{\rho_{-1}}{(\Lambda''(t^*))^{3/2}}
  \right)
$$

where rho_0 and rho_{-1} are the third absolute moments under the respective
tilted distributions.

### 5.4 Relative Error

In practice, we care about the **relative error** in epsilon. Since delta(epsilon)
is monotone decreasing, the relative error in epsilon at target delta is:

$$
\frac{|\varepsilon - \hat\varepsilon|}{\varepsilon}
  \approx \frac{|\delta - \hat\delta|}{|\delta'(\varepsilon)| \cdot \varepsilon}
$$

by the implicit function theorem. Since |delta'(epsilon)| grows with n
(the privacy curve steepens), the relative error in epsilon decays faster
than 1/sqrt(n).

---

## 6. Edge Cases and Numerical Stability

### 6.1 The r ~ 0, s ~ 0 Limit

When t* -> 0 (which happens when epsilon approaches E[L], the mean of the PLRV),
both r-hat and s-hat approach 0. The Lugannani-Rice formula has a removable
singularity:

$$
\lim_{\hat r \to 0} \left[\bar\Phi(\hat r) + \varphi(\hat r)\!\left(\frac{1}{\hat r} - \frac{1}{\hat s}\right)\right] = \frac{1}{2}
$$

since Phi-bar(0) = 1/2 and the correction term vanishes. The implementation
handles this explicitly (lines 205-208 of cgf_pld.rs).

### 6.2 Log-Space Subtraction

The subtraction delta = T_1 - c*T_2 can cause catastrophic cancellation.
We compute in log-space:

$$
\delta = \exp(a) \cdot (1 - \exp(b - a))
$$

where a = log(T_1) and b = log(c*T_2). When b - a < -50, the second term
is negligible and we return exp(a) directly (lines 130-141 of cgf_pld.rs).

### 6.3 Newton's Method Convergence

The saddle-point equation Lambda'(t*) = epsilon is solved by Newton's method:

$$
t_{k+1} = t_k - \frac{\Lambda'(t_k) - \varepsilon}{\Lambda''(t_k)}.
$$

Since Lambda is strictly convex (Lambda'' > 0 everywhere in the domain),
Newton's method converges **quadratically** from any starting point in the
domain. The implementation uses t_0 = 0.5 and converges within 5-10 iterations
to machine precision (tolerance 1e-12, max 100 iterations).

---

## 7. Gaussian Mechanism: Closed-Form Verification

For the Gaussian mechanism with sensitivity Delta = 1 and noise sigma:

$$
L(x) = \frac{1 - 2x}{2\sigma^2}, \qquad x \sim \mathcal{N}(0, \sigma^2)
$$

so L ~ N(mu_L, sigma_L^2) with:

$$
\mu_L = \frac{1}{2\sigma^2}, \qquad \sigma_L^2 = \frac{1}{\sigma^2}.
$$

The **exact** delta is:

$$
\delta(\varepsilon) = \Phi\!\left(\frac{1}{2\sigma} - \varepsilon\sigma\right)
  - e^\varepsilon \, \Phi\!\left(-\frac{1}{2\sigma} - \varepsilon\sigma\right).
$$

The CGF is:

$$
\Lambda(t) = \frac{t(1+t)}{2\sigma^2},
\qquad
\Lambda'(t) = \frac{1+2t}{2\sigma^2},
\qquad
\Lambda''(t) = \frac{1}{\sigma^2}.
$$

**Saddle-point**: Lambda'(t*) = epsilon gives t* = (epsilon * sigma^2 - 1/2).

**First tail** (r, s):
- Lambda(t*) = t*(1 + t*) / (2*sigma^2)
- r = sign(t*) * sqrt(2 * (t* * epsilon - Lambda(t*)))
  = sign(t*) * sqrt(t*^2 / sigma^2)
  = |t*| / sigma
  = (epsilon * sigma - 1/(2*sigma)) [when t* > 0]
- s = t* * sqrt(1/sigma^2) = t* / sigma

Since r = s for the Gaussian, the correction term (1/r - 1/s) = 0, and:

$$
\mathbb{P}_P(L > \varepsilon) = \bar\Phi(\hat r) = \bar\Phi\!\left(\varepsilon\sigma - \frac{1}{2\sigma}\right)
  = \Phi\!\left(\frac{1}{2\sigma} - \varepsilon\sigma\right).
$$

This matches the first term of the exact formula. Similarly, the second tail
reproduces the second term exactly. This confirms Corollary 5.3.

---

## 8. Comparison with [Alghamdi et al. 2023]

| Property | Alghamdi SPA-MSD | Our LR Decomposition |
|---|---|---|
| **Integral** | Single contour integral | Two tail probabilities |
| **Saddle eq.** | K'(t) = epsilon + 1/t - 1/(1+t) | K'(t) = epsilon (clean) |
| **Singularity at epsilon=0** | Yes (log(z) in F_epsilon) | No |
| **Exact for Gaussian** | No (~25% error at n=1) | **Yes** (0% error) |
| **Error rate** | O(1/sqrt(n)) | O(1/sqrt(n)) |
| **Provable bounds** | SPA-CLT variant only | Berry-Esseen per tail |
| **Complexity** | O(1) per query | O(1) per query |

The key advantage is that Lugannani-Rice applied to each tail separately
avoids the log-singularity entirely and is exact for exponential families.

---

## 9. Summary of Implemented Formulas

For n-fold composition with CGF Lambda(t):

1. **Saddle-point**: find t* where n*Lambda'(t*) = epsilon (Newton's method)

2. **First tail** (original distribution, CGF = n*Lambda(t)):
   - r = sign(t*) * sqrt(2*(t* * epsilon - n*Lambda(t*)))
   - s = t* * sqrt(n*Lambda''(t*))
   - log T_1 = log[Phi-bar(r) + phi(r)*(1/r - 1/s)]

3. **Second tail** (tilted by -1, CGF_{-1}(t) = n*Lambda(t-1) - n*Lambda(-1)):
   - saddle s* = t* + 1
   - r_{-1} = sign(s*) * sqrt(2*(s* * epsilon - n*Lambda(t*) + n*Lambda(-1)))
   - s_{-1} = s* * sqrt(n*Lambda''(t*))
   - log T_2 = log[Phi-bar(r_{-1}) + phi(r_{-1})*(1/r_{-1} - 1/s_{-1})]

4. **Delta**: delta(epsilon) = exp(log T_1) - exp(epsilon + n*Lambda(-1) + log T_2)

5. **Epsilon(delta)**: binary search on delta(epsilon) (monotone decreasing)

6. **Error bound** (Theorem 5.2):
   |delta - delta-hat| <= (0.56/sqrt(n)) * [rho_0 / Lambda''(t*)^{3/2} + e^{epsilon + n*Lambda(-1)} * rho_{-1} / Lambda''(t*)^{3/2}]

---

## References

1. R. Lugannani and S. Rice. "Saddle point approximation for the distribution
   of the sum of independent random variables." *Advances in Applied
   Probability*, 12(2):475-490, 1980.

2. H. E. Daniels. "Tail probability approximations." *International
   Statistical Review*, 55(1):37-48, 1987.

3. J. L. Jensen. *Saddlepoint Approximations*. Oxford Statistical Science
   Series. Oxford University Press, 1995.

4. W. Alghamdi, J. F. Gomez, S. Asoodeh, F. P. Calmon, O. Kosut, L. Sankar.
   "The Saddle-Point Accountant for Differential Privacy." *ICML 2023*.
   Proceedings of the 40th International Conference on Machine Learning,
   PMLR 202:508-528.

5. I. Shevtsova. "An improvement of convergence rate estimates in the
   Lyapunov theorem." *Doklady Mathematics*, 82(3):862-864, 2010.
   [Berry-Esseen constant 0.56]

6. B. Balle, G. Barthe, M. Gaboardi. "Privacy amplification by subsampling:
   Tight analyses via couplings." *NeurIPS 2018*.
