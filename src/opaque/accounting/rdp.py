"""RDP (Rényi Differential Privacy) accountant with convenience methods."""

import dp_accounting
import numpy as np

# Default RDP orders from JAX-Privacy
_DEFAULT_RDP_ORDERS = np.concatenate(
    (
        np.linspace(1.01, 8, num=50),
        np.arange(8, 64),
        np.linspace(65, 512, num=10, dtype=int),
    )
)


class RDPAccountant(dp_accounting.rdp.RdpAccountant):
    """RDP accountant with convenience methods for common DP-SGD patterns.

    This class extends dp_accounting's RdpAccountant with high-level methods
    for tracking privacy in DP-SGD training with different sampling strategies.

    RDP accounting is faster than PLD but may give looser bounds for some
    parameter regimes. Does not support truncated Poisson sampling.

    Args:
        orders: RDP orders to compute. Default: standard orders from JAX-Privacy
            (50 orders from 1.01 to 8, then 8 to 64, then 10 orders to 512).
        neighboring_relation: Type of adjacency for DP.
            Default: ADD_OR_REMOVE_ONE.

    Example:
        >>> accountant = RDPAccountant()
        >>>
        >>> # Track Poisson sampling steps
        >>> for step in range(1000):
        >>>     accountant.step_poisson(
        >>>         noise_multiplier=1.1,
        >>>         sample_rate=0.01,
        >>>     )
        >>>
        >>> # Get privacy spent
        >>> epsilon = accountant.get_epsilon(delta=1e-5)
        >>> print(f"Privacy: ε={epsilon:.2f}, δ=1e-5")
    """

    def __init__(
        self,
        orders: np.ndarray | None = None,
        neighboring_relation: dp_accounting.NeighboringRelation = (
            dp_accounting.NeighboringRelation.ADD_OR_REMOVE_ONE
        ),
    ):
        """Initialize RDP accountant.

        Args:
            orders: RDP orders to compute. If None, uses default orders.
            neighboring_relation: Type of adjacency for DP.
                Default: ADD_OR_REMOVE_ONE.
        """
        if orders is None:
            orders = _DEFAULT_RDP_ORDERS
        super().__init__(
            orders=orders,
            neighboring_relation=neighboring_relation,
        )
        self.steps = 0

    def step_poisson(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int = 1,
    ) -> "RDPAccountant":
        """Record Poisson sampling step(s).

        Standard Poisson sampling: each example is independently included in
        the batch with probability `sample_rate`. Batch size is random with
        expected size `sample_rate * dataset_size`.

        Args:
            noise_multiplier: Ratio of noise stddev to sensitivity.
            sample_rate: Sampling probability (batch_size / dataset_size).
            num_steps: Number of steps to record (default: 1).

        Returns:
            Self for method chaining.

        Example:
            >>> acc = RDPAccountant()
            >>> acc.step_poisson(noise_multiplier=1.1, sample_rate=0.01)
            >>> eps = acc.get_epsilon(delta=1e-5)
        """
        event = dp_accounting.PoissonSampledDpEvent(
            sampling_probability=sample_rate,
            event=dp_accounting.GaussianDpEvent(noise_multiplier),
        )
        self.compose(event, num_steps)
        self.steps += num_steps
        return self

    def step_fixed_batch(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int = 1,
    ) -> "RDPAccountant":
        """Record fixed-size batch sampling (without replacement).

        Fixed-size batch sampling selects exactly `batch_size` examples uniformly
        without replacement from the dataset. This corresponds to the "replace"
        adjacency definition in differential privacy.

        **Privacy Analysis**: The privacy analysis for fixed-size sampling without
        replacement reduces to Poisson sampling with **doubled sensitivity**. This
        is because the replace adjacency (swapping one example for another) has
        twice the sensitivity of the add-or-remove adjacency (Poisson sampling).

        Equivalently, this is Poisson sampling with half the noise multiplier.

        **Exactness**: This is an **exact** reduction for the replace adjacency.
        The resulting (ε, δ)-DP guarantee is valid under the replace adjacency,
        which is a stronger notion than add-or-remove. Therefore, it also holds
        for add-or-remove adjacency (though potentially not tight).

        Reference: Standard DP literature on different adjacency definitions.
        See also JAX-Privacy implementation in `analysis.py`.

        Args:
            noise_multiplier: Ratio of noise stddev to sensitivity (for Poisson).
            sample_rate: Batch size / dataset_size.
            num_steps: Number of steps to record (default: 1).

        Returns:
            Self for method chaining.

        Example:
            >>> acc = RDPAccountant()
            >>> # Standard mini-batch SGD (without replacement)
            >>> acc.step_fixed_batch(
            >>>     noise_multiplier=1.1,
            >>>     sample_rate=0.01,  # 100 / 10000
            >>> )
            >>> eps = acc.get_epsilon(target_delta=1e-5)

        Note:
            If you're using Poisson sampling (each example independently included
            with probability q), use `step_poisson()` instead. Most DP-SGD
            implementations use Poisson sampling for simplicity.
        """
        # Fixed batch size (replace adjacency) = Poisson with double sensitivity
        # Equivalently: Poisson with half the noise multiplier
        return self.step_poisson(
            noise_multiplier=noise_multiplier / 2.0,
            sample_rate=sample_rate,
            num_steps=num_steps,
        )
