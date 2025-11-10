# Differential Privacy Basics

## What is Differential Privacy?

Differential privacy (DP) provides mathematical guarantees that a model doesn't memorize individual training examples. This is critical when training on sensitive data like medical records, financial data, or private messages.

## DP-SGD Algorithm

The DP-SGD algorithm ([Abadi et al. 2016](https://arxiv.org/abs/1607.00133)) modifies standard SGD training:

**Standard SGD:**
```python
grads = compute_batch_gradient(model, data)
model.update(grads)
```

**DP-SGD:**
```python
per_example_grads = [compute_gradient(model, x) for x in data]
clipped_grads = [clip(g, max_norm=C) for g in per_example_grads]
noisy_grads = sum(clipped_grads) + Gaussian_noise(σ)
model.update(noisy_grads)
```

The key steps are:

1. **Per-example gradients**: Compute gradients for each training example separately
2. **Clipping**: Limit each gradient to maximum L2 norm C
3. **Noise addition**: Add calibrated Gaussian noise (scale σ)
4. **Update**: Apply the noisy, clipped gradients

## Privacy Budget

DP is parameterized by (ε, δ):

- **ε (epsilon)**: Privacy loss - smaller values mean stronger privacy
- **δ (delta)**: Probability of privacy violation

**Typical values:**
- ε ∈ [1, 10]
- δ ∈ [1e-6, 1e-5]

**Trade-offs:**
- Smaller ε → more noise → lower model accuracy
- Larger batch size → better privacy-utility trade-off
- More training steps → larger privacy budget consumed

## References

- [Abadi et al. 2016 - Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)
- [DP-SGD Tutorial](https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3)
