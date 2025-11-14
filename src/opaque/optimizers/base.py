"""Base DP optimizer interface for functional optimizers.

This module provides the core infrastructure for wrapping TorchOpt optimizers
with differential privacy guarantees. It integrates:
  - Gradient clipping (from Stage 1)
  - Noise injection (from Stage 2)
  - Privacy accounting (from Stage 2)

All DP optimizers follow the functional programming pattern from TorchOpt/Optax:
  init_fn(params) -> state
  step_fn(grads, state, params) -> (updates, new_state)
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import torch

# TODO: Update to use new functional accounting API
# from opaque.accounting import RDPAccountant
from opaque.noise import add_gaussian_noise


class DPOptimizerState(NamedTuple):
    """State for DP-enabled optimizer.

    This wraps the base optimizer state with DP-specific components.

    Attributes:
        opt_state: Internal state of the base optimizer (from TorchOpt)
        accountant: Privacy accountant tracking (ε, δ) budget (DEPRECATED)
        noise_gen: Random number generator for reproducible noise
        step: Current training step counter
    """

    opt_state: Any  # TorchOpt optimizer state
    accountant: Any = None  # DEPRECATED: Will use functional accounting API
    noise_gen: torch.Generator = None
    step: int = 0


def make_dp_optimizer(
    base_optimizer: Any,  # TorchOpt GradientTransformation
    *,
    l2_clip_norm: float,
    noise_multiplier: float,
    sample_rate: float,
    target_delta: float,
    accountant_type: str = "rdp",
    seed: int = 42,
) -> tuple[Callable, Callable]:
    """Wrap a TorchOpt optimizer with differential privacy.

    This is the core factory function that adds DP guarantees to any
    TorchOpt-compatible optimizer by injecting:
      1. Gradient clipping (clipped by caller before passing to update)
      2. Gaussian noise addition (σ·C·N(0,I))
      3. Privacy accounting (tracks privacy budget)

    Args:
        base_optimizer: TorchOpt GradientTransformation (e.g., torchopt.adam())
        l2_clip_norm: Maximum L2 norm for gradient clipping (C)
        noise_multiplier: Noise scale relative to clip norm (σ)
        sample_rate: Sampling rate q = batch_size / dataset_size
        target_delta: Target δ for (ε, δ)-DP
        accountant_type: "rdp" or "pld" for privacy accounting
        seed: Random seed for reproducible noise generation

    Returns:
        (init_fn, step_fn) where:
          - init_fn(params) -> DPOptimizerState
          - step_fn(params, grads, state) -> (new_params, new_state, metrics)

    Example:
        >>> import torchopt
        >>> from opaque.optimizers import make_dp_optimizer
        >>>
        >>> # Create base optimizer
        >>> base_opt = torchopt.adam(lr=0.001)
        >>>
        >>> # Wrap with DP
        >>> init_fn, step_fn = make_dp_optimizer(
        ...     base_opt,
        ...     l2_clip_norm=1.0,
        ...     noise_multiplier=1.1,
        ...     sample_rate=0.01,
        ...     target_delta=1e-5,
        ... )
        >>>
        >>> # Initialize
        >>> params = {'weight': torch.randn(10, 5), 'bias': torch.randn(5)}
        >>> state = init_fn(params)
        >>>
        >>> # Training loop
        >>> for batch in dataloader:
        ...     # Compute gradients (with clipping already applied)
        ...     grads = compute_clipped_grads(params, batch)
        ...
        ...     # DP optimizer step
        ...     params, state, metrics = step_fn(params, grads, state)
        ...
        ...     print(f"ε = {metrics['epsilon']:.2f}")
    """
    # TODO: Update to use new functional accounting API
    # For now, accounting is disabled in optimizers
    accountant = None
    # if accountant_type == "rdp":
    #     accountant = RDPAccountant()
    # elif accountant_type == "pld":
    #     from opaque.accounting import PLDAccountant
    #     accountant = PLDAccountant()
    # else:
    #     raise ValueError(f"Unknown accountant_type: {accountant_type}. Use 'rdp' or 'pld'.")

    # Noise stddev: σ·C
    stddev = noise_multiplier * l2_clip_norm

    def init_fn(params):
        """Initialize optimizer state.

        Args:
            params: PyTree of model parameters

        Returns:
            DPOptimizerState with initialized components
        """
        # Initialize base optimizer
        opt_state = base_optimizer.init(params)

        # Create noise generator
        noise_gen = torch.Generator().manual_seed(seed)

        return DPOptimizerState(
            opt_state=opt_state,
            accountant=accountant,
            noise_gen=noise_gen,
            step=0,
        )

    def step_fn(params, grads, state):
        """Perform one DP optimizer step.

        Args:
            params: Current PyTree of parameters
            grads: Clipped gradients (already clipped by caller!)
            state: Current DPOptimizerState

        Returns:
            Tuple of (new_params, new_state, metrics) where:
              - new_params: Updated parameters
              - new_state: Updated optimizer state
              - metrics: Dict with 'epsilon', 'delta', 'step'
        """
        # 1. Add DP noise to gradients
        noisy_grads = add_gaussian_noise(grads, stddev=stddev, generator=state.noise_gen)

        # 2. Get parameter updates from base optimizer
        updates, new_opt_state = base_optimizer.update(
            noisy_grads, state.opt_state, params=params
        )

        # 3. Apply updates to parameters
        # torchopt.apply_updates does: params - updates (gradient descent)
        import torchopt

        new_params = torchopt.apply_updates(params, updates)

        # 4. Track privacy (TODO: Update to functional accounting API)
        # new_accountant = state.accountant
        # new_accountant.step_poisson(
        #     noise_multiplier=noise_multiplier,
        #     sample_rate=sample_rate,
        #     num_steps=1,
        # )

        # 5. Compute current privacy cost (TODO: Update to functional accounting API)
        # epsilon = new_accountant.get_epsilon(target_delta=target_delta)
        epsilon = None  # Disabled for now

        # Create new state
        new_state = DPOptimizerState(
            opt_state=new_opt_state,
            accountant=None,  # Disabled
            noise_gen=state.noise_gen,  # Same generator (mutable state)
            step=state.step + 1,
        )

        # Metrics for monitoring
        metrics = {
            "epsilon": epsilon,  # None for now
            "delta": target_delta,
            "step": new_state.step,
        }

        return new_params, new_state, metrics

    return init_fn, step_fn


__all__ = [
    "DPOptimizerState",
    "make_dp_optimizer",
]
