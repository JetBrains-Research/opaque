# Rectified Gaussian Mechanism

The rectified Gaussian mechanism samples noise from a standard Gaussian and
**clamps** the result to a bounded interval $[-R\sigma, R\sigma]$. The
probability mass that would have fallen outside the bounds accumulates as
point masses at the boundaries. This bounded support gives tighter privacy
accounting than the standard Gaussian at the same noise level.

## Idea

Standard Gaussian noise has unbounded support — with vanishingly small
probability, it can produce enormous values. While this probability is tiny,
it forces the privacy analysis to account for worst-case outcomes, loosening
the $\varepsilon$ bound.

The rectified mechanism eliminates these tails by clamping: any noise sample
outside $[-R\sigma, R\sigma]$ is snapped to the nearest boundary. The
resulting distribution is a mixture of:

- A **continuous part** (the Gaussian density restricted to the interior)
- **Point masses** at $\pm R\sigma$ (the accumulated tail probability)

Because the support is bounded, the Data Processing Inequality (DPI) guarantees
that the rectified mechanism is at least as private as the standard Gaussian.
In practice, the improvement is 5–15% tighter $\varepsilon$ for typical
radius values ($R = 3$–$10$).

**When to use**: When you want tighter accounting without changing the noise
level. The implementation is trivial (just clamp), and the accounting
improvement is free. However, the [truncated Gaussian](truncated-gaussian.md)
is always at least as tight — prefer it when possible.

## Mathematics

### Noise distribution

The rectified Gaussian has a mixed distribution. For a query with true answer
$\mu$:

**Interior** ($x \in (-R\sigma, R\sigma)$):

$$f_\mu(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

**Left boundary** (point mass at $-R\sigma$):

$$p_L(\mu) = \Phi\!\left(\frac{-R\sigma - \mu}{\sigma}\right)$$

**Right boundary** (point mass at $+R\sigma$):

$$p_R(\mu) = 1 - \Phi\!\left(\frac{R\sigma - \mu}{\sigma}\right)$$

where $\Phi$ is the standard normal CDF.

### Privacy loss function

In the interior, the privacy loss is the same as for the standard Gaussian:

$$\ell(x) = \frac{\Delta}{\sigma^2}\!\left(\frac{\Delta}{2} - x\right)$$

At the boundaries, the privacy loss involves the ratio of point masses
(or point mass to continuous density), which can be larger than interior
values.

### Hockey-stick divergence

The total divergence $\delta(\varepsilon)$ splits into three parts:

$$\delta(\varepsilon) = \underbrace{\int_{-R\sigma}^{R\sigma} \max(0,\, f_0(x) - e^\varepsilon f_\Delta(x))\, dx}_{\text{interior}} + \underbrace{\max(0,\, p_L(0) - e^\varepsilon p_L(\Delta))}_{\text{left mass}} + \underbrace{\max(0,\, p_R(0) - e^\varepsilon p_R(\Delta))}_{\text{right mass}}$$

### Key properties

- **DPI guarantee**: $\varepsilon_{\text{rectified}} \leq \varepsilon_{\text{Gaussian}}$ at all $\delta$
- **Larger radius → closer to Gaussian**: as $R \to \infty$, the point masses vanish and the mechanism converges to the standard Gaussian
- **Higher $\sigma$ → lower $\varepsilon$**: more noise always improves privacy

## Supported amplifications

### Poisson subsampling (`poisson`)

Works with standard Poisson subsampling. A key mathematical identity
simplifies the analysis: when the crossover point $x_{\text{cut}}$ (where
the privacy loss equals $\varepsilon$) lies inside the domain $[-R\sigma, R\sigma]$,
the rectified $\delta(\varepsilon)$ **exactly equals** the standard Poisson
Gaussian $\delta(\varepsilon)$. This happens because the point mass compensation
cancels the truncated tails:

$$\Phi(h_i/\sigma) - \Phi(l_o/\sigma) + p_{\text{mass}} = 1 - \Phi(l_o/\sigma)$$

In practice, for reasonable $R$ ($\geq 3$), the crossover is almost always inside
the domain, so the amplified analysis delegates to the standard Poisson Gaussian
code path for numerical stability.

```python
step = acc.poisson(acc.rectified_gaussian(1.0, radius=5.0), sample_rate=0.01)
training = step * 1000
eps = training.epsilon_at(delta=1e-5)
```

| Amplification | Supported | Notes |
|---------------|:---------:|-------|
| `poisson()` | Yes | Tight analysis via crossover identity |
| `truncated_poisson()` | No | Use standard Gaussian instead |
| `cyclic_poisson()` | No | For MF mechanisms only |

## Code examples

### Noise injection

```python
from opaque.noise import rectified_gaussian_noise
from opaque.random import key

noise_fn, noise_state = rectified_gaussian_noise(
    stddev=noise_multiplier * clip_state.sensitivity(),
    radius=5.0,          # clamp to ±5σ
    key=key(42),
)

for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads
```

The `radius` parameter is in multiples of the standard deviation $\sigma$
(the same convention used by the accounting API). For example, `radius=5.0`
clamps the noise to the interval $[-5\sigma, 5\sigma]$ in both the noise
injection and accounting calls.

### Privacy accounting

```python
import opaque.accounting as acc

# Single step with Poisson subsampling
step = acc.poisson(
    acc.rectified_gaussian(noise_multiplier=1.0, radius=5.0),
    sample_rate=0.01,
)
training = step * 1000
eps = training.epsilon_at(delta=1e-5)

# Compare with standard Gaussian
step_gauss = acc.poisson(acc.gaussian(1.0), sample_rate=0.01)
eps_gauss = (step_gauss * 1000).epsilon_at(delta=1e-5)
print(f"Rectified: ε={eps:.4f},  Gaussian: ε={eps_gauss:.4f}")
```

### Calibration

```python
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(
        acc.rectified_gaussian(nm, radius=5.0), sample_rate=0.01
    ) * 1000,
    param_min=0.1,
    param_max=10.0,
)
noise_multiplier = result.param
```

## Parameter guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| `noise_multiplier` | 0.1 – 10.0 (calibrate) | Higher = more private. Use `acc.calibrate()`. |
| `radius` | 3.0 – 10.0 | Bounds in sigma units. Smaller = tighter $\varepsilon$ but more mass at boundaries. |

**Tips**:

- **Radius 5.0** is a good default — captures 99.99994% of the Gaussian mass
  (only $6 \times 10^{-7}$ is clipped per tail).
- **Radius 3.0** clips ~0.13% of mass per tail, giving a more noticeable
  privacy improvement but also more point-mass artifacts.
- **Radius > 10** gives negligible improvement over standard Gaussian.
- If you want the tightest bounds, use
  [Truncated Gaussian](truncated-gaussian.md) instead — it is always at
  least as tight as rectified.

## References

- **Hu, Zheng, Li (2024)** — [Bounded Gaussian Mechanism for Differential Privacy](https://arxiv.org/abs/2403.05598).
  Analysis of rectified and truncated Gaussian mechanisms.
- **Chen & Hale (2024)** — [The Bounded Gaussian Mechanism for Differential Privacy](https://arxiv.org/abs/2211.17230).
  Journal of Privacy and Confidentiality, 14(1). Original bounded Gaussian
  mechanism with inverse-CDF sampling.
