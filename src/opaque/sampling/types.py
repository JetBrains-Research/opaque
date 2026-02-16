"""Types and enums for sampling strategies."""

from enum import Enum


class SamplingMode(Enum):
    """Sampling strategy for distributed training.

    Two modes provide different privacy accounting guarantees and
    are suitable for different training scenarios.
    """

    INDEPENDENT = "independent"
    """Each worker samples independently with the same sample_rate.
    
    **Privacy Accounting:** Leads to mixture Gaussian accounting. The privacy
    analysis for this mode is more complex and is planned for future work.
    
    **Use Case:** 
    - Backward compatibility with existing code
    - When future accounting module supports mixture Gaussian composition
    - Single-device training (default)
    
    **Distributed Behavior:**
    - Each worker uses its own RNG
    - Workers may get different batch sizes (expected from Poisson)
    - No synchronization required
    
    **Warning:** Using this mode in distributed training without proper mixture
    Gaussian accounting may underestimate privacy cost.
    """

    SHARDED = "sharded"
    """Workers partition dataset and sample from shards independently.
    
    **Implementation: "Single Poisson via partitioning"**
    - Dataset partitioned across workers (disjoint shards)
    - Each worker independently samples its shard with probability p
    - Mathematical property: Poisson(p) on union of disjoint sets = union of Poisson(p) on each set
    - This ensures "single Poisson" execution for correct DP-SGD accounting
    
    **Privacy Accounting:** Standard Gaussian accounting applies. Use
    `compose_poisson_gaussian()` for privacy tracking. Mathematically equivalent
    to running single Poisson on full dataset.
    
    **Use Case:**
    - **All distributed training scenarios** (default for world_size > 1)
    - Single-machine multi-GPU (dataset fits in memory or partitioned)
    - Multi-node distributed training with partitioned data
    - Federated learning scenarios
    - Most scalable approach for production
    
    **Distributed Behavior:**
    - Each worker samples only from its assigned shard
    - Workers use independent RNGs (but on disjoint data!)
    - Requires `rank` and `world_size` parameters for partitioning
    - Dataset is partitioned: worker i samples from indices
      [i*n/k, (i+1)*n/k) where n=dataset size, k=world_size
    - **Zero communication overhead** - no synchronization needed
    
    **Advantages:**
    - Works with any dataset size (including too large for single worker)
    - No network communication required
    - Most scalable for production deployments
    - Covers all practical distributed training use cases
    
    **Limitation:** Requires dataset to be consistently partitioned across
    workers. Uneven partitioning may affect utility.
    """


__all__ = ["SamplingMode"]
