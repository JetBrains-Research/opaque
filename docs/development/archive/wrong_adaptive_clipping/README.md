# Wrong Adaptive Clipping Implementation (Archived)

**Date Archived**: 2026-02-12
**Reason**: Implemented wrong algorithm

---

## What Was This?

This was an attempt to implement adaptive clipping based on:
- **Zuo et al. 2024 (DP-Adam-AC)**: https://arxiv.org/abs/2510.05288
- Included: Percentile-based adaptive clipping + LR scaling + EMA

## Why Was It Wrong?

We needed **Andrew et al. 2021** (https://arxiv.org/abs/1905.03871):
- Uses **geometric updates**: `C_{t+1} = C_t * exp(η * sign(ρ_t - γ))`
- Updates based on **clipping rate** (fraction of gradients clipped)
- This is the **foundational** validated algorithm

What we implemented instead:
- **Percentile-based**: `C = Percentile_q(buffer)` where `q = 1 - target_clip_rate`
- This is a heuristic, not the Andrew et al. algorithm
- Mixed clipping logic with optimizer logic (should be separated)

## Archived Files

- `dp_optimizer_ac.py` - Optimizer wrapper with wrong adaptive clipping
- `adaptive/clip_buffer.py` - Percentile-based buffer (not geometric updates)
- `adaptive/lr_scheduler.py` - LR scaling based on clip rate
- `types.py` - State types for the optimizer

## What Replaced This?

- **`src/opaque/clipping/adaptive.py`** - Correct Andrew et al. 2021 implementation
- Uses `adaptive_clipped_grad()` function
- Geometric updates with proper algorithm
- Clean separation: clipping is separate from optimization

## Could This Be Useful?

The percentile-based approach might be interesting as an **alternative** adaptive clipping strategy, but:
1. Not validated in literature
2. Not the standard Andrew et al. algorithm
3. Would need to be clearly marked as experimental

If we want to resurrect this, it should be:
- Clearly separated from the standard Andrew et al. implementation
- Marked as experimental/alternative
- In a separate module like `experimental_adaptive.py`

---

**Bottom line**: We implemented Zuo et al. (complex, unvalidated) when we needed Andrew et al. (simple, foundational, validated).
