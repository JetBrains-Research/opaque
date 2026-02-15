"""Poisson samplers for differential privacy.

These samplers implement Poisson subsampling, where each example in the dataset
is independently included in a batch with probability `sample_rate`. This provides
privacy amplification, reducing the privacy cost compared to fixed-batch sampling.

For distributed training, set `distributed=True` to ensure all devices use the
same sampled indices. Rank 0 samples and broadcasts indices to all other devices.
"""

from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


class PoissonSampler(Sampler):
    """Poisson sampler for privacy amplification.

    Each example in the dataset is independently included with probability
    `sample_rate`. This creates variable-sized batches, which provides privacy
    amplification: the effective privacy cost is reduced by approximately
    √(1/sample_rate) compared to uniform sampling.

    Args:
        data_source: Dataset to sample from (any object with __len__)
        sample_rate: Probability of including each example (0 < p <= 1)
        num_epochs: Number of epochs to iterate over
        generator: Optional numpy random generator for reproducibility
        distributed: If True, ensure all devices use same sampled indices.
            Rank 0 samples and broadcasts to all devices. Requires
            torch.distributed to be initialized. Default: False.

    Example:
        >>> dataset = MyDataset(...)
        >>> sampler = PoissonSampler(dataset, sample_rate=0.01, num_epochs=10)
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
        >>>
        >>> for batch in loader:
        ...     # batch has variable size!
        ...     # Expected size: len(dataset) * sample_rate
        ...     pass

    Example with distributed training:
        >>> import torch.distributed as dist
        >>> from opaque.sampling import PoissonSampler
        >>>
        >>> # Initialize distributed
        >>> dist.init_process_group(backend='nccl')
        >>>
        >>> # Create sampler with distributed=True
        >>> dataset = MyDataset(...)
        >>> sampler = PoissonSampler(
        ...     dataset,
        ...     sample_rate=0.01,
        ...     num_epochs=10,
        ...     distributed=True,  # Rank 0 samples, broadcasts to all
        ... )
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
        >>>
        >>> # All devices get same sampled indices
        >>> for batch in loader:
        ...     # Same batch on all devices (for privacy accounting)
        ...     pass

    Note:
        - Batch sizes are variable (Poisson property)
        - Expected batch size: len(dataset) * sample_rate
        - Variance: len(dataset) * sample_rate * (1 - sample_rate)
        - Use with DataLoader's batch_sampler parameter (not sampler)
    """

    def __init__(
        self,
        data_source,
        sample_rate: float,
        num_epochs: int = 1,
        generator: np.random.Generator | None = None,
        distributed: bool = False,
    ):
        super().__init__()

        if not 0 < sample_rate <= 1:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
        if num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")

        self.data_source = data_source
        self.sample_rate = sample_rate
        self.num_epochs = num_epochs
        self.generator = generator if generator is not None else np.random.default_rng()
        self.distributed = distributed

        self._num_samples = len(data_source)

        # Check distributed initialization if needed
        if self.distributed:
            try:
                from opaque.distributed import is_initialized

                if not is_initialized():
                    raise RuntimeError(
                        "distributed=True but torch.distributed is not initialized. "
                        "Call torch.distributed.init_process_group() first."
                    )
            except ImportError:
                raise RuntimeError(
                    "distributed=True but opaque.distributed module not available"
                ) from None

    def __iter__(self) -> Iterator[list[int]]:
        """Yield variable-size batches as lists of indices.

        Each call samples the entire dataset once per epoch using Poisson
        subsampling. Examples are included independently with probability
        sample_rate.

        In distributed mode, rank 0 samples and broadcasts indices to all devices.

        Returns:
            Iterator yielding lists of indices (variable size)
        """
        for _ in range(self.num_epochs):
            if self.distributed:
                # Distributed: rank 0 samples, broadcasts to all
                import torch.distributed as dist

                from opaque.distributed import get_rank

                rank = get_rank()

                # Rank 0: sample indices
                if rank == 0:
                    included = (
                        self.generator.random(self._num_samples) < self.sample_rate
                    )
                    indices = np.where(included)[0]
                    batch_size = len(indices)
                else:
                    indices = None
                    batch_size = 0

                # Broadcast batch size
                batch_size_tensor = torch.tensor(batch_size, dtype=torch.long)
                dist.broadcast(batch_size_tensor, src=0)
                batch_size = batch_size_tensor.item()

                # Allocate buffer on all ranks
                if rank != 0:
                    indices = np.zeros(batch_size, dtype=np.int64)

                # Broadcast indices if batch is non-empty
                if batch_size > 0:
                    indices_tensor = torch.from_numpy(indices).long()
                    dist.broadcast(indices_tensor, src=0)
                    indices = indices_tensor.numpy()

                # Yield as list
                yield indices.tolist()
            else:
                # Single-device: standard poisson sampling
                included = self.generator.random(self._num_samples) < self.sample_rate
                indices = np.where(included)[0]
                yield indices.tolist()

    def __len__(self) -> int:
        """Return number of batches (one per epoch)."""
        return self.num_epochs

    @property
    def expected_batch_size(self) -> float:
        """Expected batch size = num_samples * sample_rate."""
        return self._num_samples * self.sample_rate

    @property
    def batch_size_variance(self) -> float:
        """Variance of batch size for Poisson sampling."""
        return self._num_samples * self.sample_rate * (1 - self.sample_rate)
