# Truncated Gaussian Mechanism

The truncated Gaussian mechanism samples noise from a Gaussian
**renormalized** over a bounded interval $[-R\sigma, R\sigma]$. Unlike
the [rectified variant](rectified-gaussian.md) which clamps, truncation
resamples: the density is smoothly rescaled so that no probability mass
sits at the boundaries. This gives the **tightest privacy bounds** among
all three Gaussian variants.

## Idea

The key insight is that a renormalized density has no point masses and no
discontinuities. The privacy loss function is linear in $x$ (just like
standard Gaussian) but shifted by a normalization constant ratio. Because
the density is smoother than the rectified variant, the hockey-stick
divergence is smaller at every $\varepsilon$.

In practice, the truncated mechanism gives 5–15% tighter $\varepsilon$
than standard Gaussian and is always at least as tight as rectified
Gaussian, with no additional computational cost at training time (the
inverse-CDF sampling is negligibly more expensive than Gaussian sampling).

**When to use**: Whenever you would use the standard Gaussian mechanism.
The truncated variant is strictly better for privacy accounting at the
same noise level. The only trade-off is a slightly more complex noise
sampling procedure (inverse-CDF instead of direct sampling).

## Mathematics

### Noise distribution

The truncated Gaussian density on $[-R\sigma, R\sigma]$, centered at $\mu$:

$$f_\mu(x) = \frac{\varphi\!\left(\frac{x - \mu}{\sigma}\right)}{\sigma \cdot Z(\mu)}$$

where $\varphi$ is the standard normal PDF and $Z(\mu)$ is the normalization
constant:

$$Z(\mu) = \Phi\!\left(\frac{R\sigma - \mu}{\sigma}\right) - \Phi\!\left(\frac{-R\sigma - \mu}{\sigma}\right)$$

For $R \geq 3$, $Z(\mu) \approx 1$ and the density is nearly identical to the
standard Gaussian.

### Sampling procedure (inverse CDF)

For each element with value $c$:

1. Compute $\alpha = \Phi\!\left(\frac{-R\sigma - c}{\sigma}\right)$ and $\beta = \Phi\!\left(\frac{R\sigma - c}{\sigma}\right)$
2. Draw $u \sim \text{Uniform}(\alpha, \beta)$
3. Return $c + \sigma\sqrt{2}\;\text{erfinv}(2u - 1)$

This guarantees all samples lie in $[-R\sigma, R\sigma]$ without rejection.

### Privacy loss function

$$\ell(x) = \log \frac{f_0(x)}{f_\Delta(x)} = -\frac{\Delta \cdot x}{\sigma^2} + \frac{\Delta^2}{2\sigma^2} + \log \frac{Z(\Delta)}{Z(0)}$$

This is **linear in $x$** with slope $-\Delta/\sigma^2$ — the same slope as the
standard Gaussian, but shifted by the normalization ratio $\log(Z(\Delta)/Z(0))$.
For large $R$, $Z(\Delta) \approx Z(0) \approx 1$ and the shift vanishes.

### Hockey-stick divergence

The crossover point where $\ell(x) = \varepsilon$ is:

$$x_{\text{cross}} = \frac{\sigma^2}{\Delta}\!\left(\frac{\Delta^2}{2\sigma^2} + \log \frac{Z(\Delta)}{Z(0)} - \varepsilon\right)$$

and $\delta(\varepsilon) = \int_{-R\sigma}^{x_{\text{cross}}} \left(f_0(x) - e^\varepsilon f_\Delta(x)\right) dx$ (when positive).

### Privacy ordering

For any noise multiplier $\sigma$ and radius $R$:

$$\varepsilon_{\text{truncated}}(\delta) \leq \varepsilon_{\text{rectified}}(\delta) \leq \varepsilon_{\text{Gaussian}}(\delta)$$

The truncated mechanism is strictly tighter than rectified because:

1. No point masses means the density is continuous everywhere
2. The density is nonzero throughout the interior (no "lumps" at boundaries)
3. The log-density ratio is smoother, giving a smaller hockey-stick integral

## Supported amplifications

### Poisson subsampling (`poisson`)

Works with standard Poisson subsampling. When $Z_0 \approx Z_1 \approx 1$ (large
radius), the implementation delegates to the standard Poisson Gaussian code path
for numerical stability. For smaller radius, dedicated PLD computation handles
the normalization constants correctly.

```python
step = acc.poisson(acc.truncated_gaussian(1.0, radius=5.0), sample_rate=0.01)
training = step * 1000
eps = training.pmf().epsilon_at(delta=1e-5)  # tightest of the three
```

**Full ordering with Poisson amplification**:

$$\varepsilon_{\text{Poisson-truncated}} \leq \varepsilon_{\text{Poisson-rectified}} \leq \varepsilon_{\text{Poisson-Gaussian}}$$

| Amplification | Supported | Notes |
|---------------|:---------:|-------|
| `poisson()` | Yes | Full ordering preserved under Poisson |
| `truncated_poisson()` | No | Use standard Gaussian instead |
| `cyclic_poisson()` | No | For MF mechanisms only |

## Code examples

### Noise injection

```python
from opaque import truncated_gaussian_noise
from opaque.random import key

# bounds = (-radius * stddev, radius * stddev) in absolute units
stddev = noise_multiplier * clip_state.sensitivity()
bound = 5.0 * stddev  # radius=5.0 in sigma units

noise_fn, noise_state = truncated_gaussian_noise(
    stddev=stddev,
    bounds=(-bound, bound),
    key=key(42),
)

for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads
```

!!! note "Bounds vs radius"
    The noise function takes absolute `bounds=(-B, B)` while the accounting
    takes `radius` in sigma units. To match: set `B = radius * stddev`.

### Privacy accounting

```python
import opaque.accounting as acc

# Truncated Gaussian with Poisson subsampling
step = acc.poisson(
    acc.truncated_gaussian(noise_multiplier=1.0, radius=5.0),
    sample_rate=0.01,
)
training = step * 1000
eps = training.pmf().epsilon_at(delta=1e-5)

# Compare all three variants at the same noise level
for name, mech in [
    ("Gaussian",  acc.gaussian(1.0)),
    ("Rectified", acc.rectified_gaussian(1.0, radius=5.0)),
    ("Truncated", acc.truncated_gaussian(1.0, radius=5.0)),
]:
    proc = acc.poisson(mech, sample_rate=0.01) * 1000
    print(f"{name:10s}  ε = {proc.pmf().epsilon_at(1e-5):.4f}")
```

### Calibration

```python
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(
        acc.truncated_gaussian(nm, radius=5.0), sample_rate=0.01
    ) * 1000,
    param_min=0.1,
    param_max=10.0,
)
noise_multiplier = result.param
```

Because truncated accounting is tighter, calibration returns a **smaller
noise multiplier** for the same target $\varepsilon$ — meaning less noise
and better utility.

## Parameter guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| `noise_multiplier` | 0.1 – 10.0 (calibrate) | Higher = more private. Use `acc.calibrate()`. |
| `radius` | 3.0 – 10.0 | Bounds in sigma units. Smaller = tighter $\varepsilon$. |

**Tips**:

- **Radius 5.0** is the recommended default. Negligible truncation effect
  on gradients but meaningful privacy improvement.
- **Radius 3.0** gives the most privacy improvement (~10-15% tighter
  $\varepsilon$) but truncates ~0.27% of total noise mass. Verify that
  gradient quality is not affected.
- **Truncated is always better than rectified** — there is no scenario
  where you would prefer rectified over truncated for the same radius.
  The only reason to use rectified is if your noise implementation uses
  simple clamping and you want matching accounting.
- The `bounds` parameter in the noise function uses absolute units. Convert
  from sigma units: `bounds = (-radius * stddev, radius * stddev)`.

## References

- **Hu, Zheng, Li (2024)** — [Bounded Gaussian Mechanism for Differential Privacy](https://arxiv.org/abs/2403.05598).
  Privacy analysis of both rectified and truncated variants.
- **Chen & Hale (2024)** — [The Bounded Gaussian Mechanism for Differential Privacy](https://arxiv.org/abs/2211.17230).
  Journal of Privacy and Confidentiality, 14(1). Original truncated Gaussian
  mechanism using inverse-CDF sampling.
