# Stage 3: Functional Optimizers & DP-SGD Integration

**Goal**: Implement DP-SGD training loop with functional optimizers using TorchOpt

**Timeline**: 3-4 weeks

**Status**: 📋 Ready to start

---

## Overview

Stage 3 builds on the clipping (Stage 1) and noise+accounting (Stage 2) foundations to create complete DP-SGD training
workflows. We'll use **TorchOpt** as the functional optimizer library and wrap it with DP-specific functionality.

### Why TorchOpt?

TorchOpt provides JAX-like functional optimizers for PyTorch:

- **Functional API**: Stateless optimizer updates matching our functional design
- **Similar to Optax**: Mirrors JAX-Privacy's optimizer pattern
- **Efficient**: CPU/GPU acceleration and tree operations
- **Standard optimizers**: SGD, Adam, AdamW, RMSprop

### Deliverables

1. **`opaque.optimizers`** (~200 LOC)
  - `DPOptimizerState` - Lightweight state container
  - `make_dp_optimizer()` - Wrap TorchOpt optimizers with DP
  - Integration with clipping, noise, and accounting
  - Support for SGD, Adam, AdamW

2. **`opaque.training`** (~150 LOC)
  - `dp_sgd_step()` - Single DP-SGD training step
  - `create_train_state()` - Initialize training state
  - Helper functions for common patterns

3. **Tutorial 03** (Jupyter notebook)
  - Complete DP-SGD training example
  - Comparison with non-private baseline
  - Visualization of privacy-utility tradeoff

4. **Tests** (~300 LOC)
  - Optimizer state management
  - DP-SGD step correctness
  - End-to-end training validation
  - JAX-Privacy numerical equivalence

---

## Week 1: TorchOpt Integration & DP Optimizer Wrapper

### Days 1-2: TorchOpt Setup & Exploration

**File**: `pyproject.toml`

**Tasks**:

1. Add TorchOpt dependency
   ```toml
   [project]
   dependencies = [
       # ... existing
       "torchopt>=0.7.3",
   ]
   ```

2. Explore TorchOpt functional API
   ```python
   import torchopt

   # Functional optimizer example
   params = torch.randn(10, requires_grad=True)
   optimizer = torchopt.sgd(lr=0.01)
   opt_state = optimizer.init(params)

   # Update step
   loss = (params ** 2).sum()
   grads = torch.autograd.grad(loss, params)[0]
   updates, opt_state = optimizer.update(grads, opt_state)
   params = torchopt.apply_updates(params, updates)
   ```

3. Understand TorchOpt patterns:
  - `optimizer.init(params)` → state
  - `optimizer.update(grads, state)` → (updates, new_state)
  - `torchopt.apply_updates(params, updates)` → new_params
  - PyTree support for parameters

**Tests to write** (`tests/optimizers/test_torchopt_integration.py`):

```python
def test_torchopt_sgd_basic():
  """Verify TorchOpt SGD works with simple problem."""
  params = {'weight': torch.randn(10, 5), 'bias': torch.randn(10)}
  optimizer = torchopt.sgd(lr=0.1)
  opt_state = optimizer.init(params)

  # Dummy gradients
  grads = {'weight': torch.randn(10, 5), 'bias': torch.randn(10)}

  # Update
  updates, new_state = optimizer.update(grads, opt_state)
  new_params = torchopt.apply_updates(params, updates)

  assert set(new_params.keys()) == set(params.keys())
  assert not torch.equal(new_params['weight'], params['weight'])


def test_torchopt_adam_pytree():
  """Verify TorchOpt Adam works with nested PyTrees."""
  params = {
    'layer1': {'weight': torch.randn(10, 5), 'bias': torch.randn(10)},
    'layer2': {'weight': torch.randn(5, 3), 'bias': torch.randn(5)},
  }

  optimizer = torchopt.adam(lr=0.001)
  opt_state = optimizer.init(params)

  # Create matching gradient structure
  grads = tree_map(lambda p: torch.randn_like(p), params)

  updates, new_state = optimizer.update(grads, opt_state)
  new_params = torchopt.apply_updates(params, updates)

  # Verify structure preserved
  assert set(new_params.keys()) == set(params.keys())
  assert set(new_params['layer1'].keys()) == {'weight', 'bias'}
```

### Days 3-5: DP Optimizer Wrapper

**File**: `src/opaque/optimizers/__init__.py`, `src/opaque/optimizers/dp_optimizer.py`

**Goal**: Wrap TorchOpt optimizers with DP-SGD functionality

**Implementation**:

```python
"""DP-aware functional optimizers built on TorchOpt."""

from typing import Any, Callable, Optional, NamedTuple
import torch
import torchopt
from opaque.clipping import clipped_grad, BoundedSensitivityCallable
from opaque.noise import add_gaussian_noise
from opaque.accounting import PLDAccountant, RDPAccountant


class DPOptimizerState(NamedTuple):
  """State for DP optimizer.

  Attributes:
      opt_state: TorchOpt optimizer state
      accountant: Privacy accountant
      noise_generator: Random generator for noise
      step: Current training step
  """
  opt_state: Any  # TorchOpt state (opaque)
  accountant: Any  # PLDAccountant or RDPAccountant
  noise_generator: torch.Generator
  step: int


def make_dp_optimizer(
  optimizer: str | Callable = "sgd",
  learning_rate: float = 0.01,
  l2_clip_norm: float = 1.0,
  noise_multiplier: float = 1.0,
  sample_rate: float = 0.01,
  accountant_type: str = "rdp",
  target_delta: float = 1e-5,
  seed: int = 42,
  **optimizer_kwargs,
) -> tuple[Callable, Callable, Callable]:
  """Create DP-SGD optimizer from TorchOpt optimizer.

  Returns three functions for the complete DP-SGD workflow:
  1. `init_fn(params)` - Initialize optimizer state
  2. `step_fn(params, state, loss_fn, batch)` - Single DP-SGD step
  3. `get_privacy_spent(state)` - Get current (ε, δ)

  Args:
      optimizer: TorchOpt optimizer name ("sgd", "adam", "adamw") or callable
      learning_rate: Learning rate for optimizer
      l2_clip_norm: Clipping norm for gradients
      noise_multiplier: Noise stddev = noise_multiplier × clip_norm
      sample_rate: Batch size / dataset size (for accounting)
      accountant_type: "rdp" or "pld" for privacy accounting
      target_delta: Target delta for (ε, δ)-DP
      seed: Random seed for noise generation
      **optimizer_kwargs: Additional arguments for TorchOpt optimizer

  Returns:
      (init_fn, step_fn, get_privacy_spent_fn)

  Example:
      >>> # Create DP-SGD optimizer
      >>> init_fn, step_fn, get_privacy = make_dp_optimizer(
      ...     optimizer="sgd",
      ...     learning_rate=0.01,
      ...     l2_clip_norm=1.0,
      ...     noise_multiplier=1.1,
      ...     sample_rate=0.01,
      ... )
      >>>
      >>> # Initialize
      >>> params = {'weight': torch.randn(10, 5), 'bias': torch.randn(10)}
      >>> state = init_fn(params)
      >>>
      >>> # Training loop
      >>> for batch in dataloader:
      ...     params, state, loss = step_fn(params, state, loss_fn, batch)
      ...     if step % 100 == 0:
      ...         eps, delta = get_privacy(state)
      ...         print(f"Step {step}: loss={loss:.4f}, ε={eps:.2f}")
  """
  # 1. Create base TorchOpt optimizer
  if isinstance(optimizer, str):
    optimizer_map = {
      "sgd": torchopt.sgd,
      "adam": torchopt.adam,
      "adamw": torchopt.adamw,
      "rmsprop": torchopt.rmsprop,
    }
    if optimizer not in optimizer_map:
      raise ValueError(f"Unknown optimizer: {optimizer}. Choose from {list(optimizer_map.keys())}")
    base_optimizer = optimizer_map[optimizer](lr=learning_rate, **optimizer_kwargs)
  else:
    # Custom TorchOpt optimizer
    base_optimizer = optimizer(lr=learning_rate, **optimizer_kwargs)

  # 2. Create privacy accountant
  if accountant_type == "rdp":
    accountant = RDPAccountant()
  elif accountant_type == "pld":
    accountant = PLDAccountant()
  else:
    raise ValueError(f"Unknown accountant_type: {accountant_type}")

  # 3. Initialize function
  def init_fn(params):
    """Initialize DP optimizer state.

    Args:
        params: PyTree of model parameters

    Returns:
        DPOptimizerState
    """
    opt_state = base_optimizer.init(params)
    noise_gen = torch.Generator().manual_seed(seed)

    return DPOptimizerState(
      opt_state=opt_state,
      accountant=accountant,
      noise_generator=noise_gen,
      step=0,
    )

  # 4. Step function (the core DP-SGD logic)
  def step_fn(
    params,
    state: DPOptimizerState,
    loss_fn: Callable,
    batch: Any,
  ) -> tuple[Any, DPOptimizerState, float]:
    """Perform single DP-SGD training step.

    Args:
        params: Current parameters (PyTree)
        state: Current DP optimizer state
        loss_fn: Loss function with signature loss_fn(params, batch) -> scalar
        batch: Training batch data

    Returns:
        (new_params, new_state, loss_value)
    """
    # 1. Create clipped gradient function
    clipped_grad_fn = clipped_grad(
      loss_fn,
      argnums=0,  # w.r.t. params
      batch_argnums=1,  # batch is batched
      l2_clip_norm=l2_clip_norm,
    )

    # 2. Compute clipped gradients (summed over batch)
    clipped_grads = clipped_grad_fn(params, batch)

    # 3. Add Gaussian noise
    stddev = noise_multiplier * l2_clip_norm
    noisy_grads = add_gaussian_noise(
      clipped_grads,
      stddev=stddev,
      generator=state.noise_generator,
    )

    # 4. Apply optimizer update
    updates, new_opt_state = base_optimizer.update(
      noisy_grads,
      state.opt_state,
      params=params,
    )
    new_params = torchopt.apply_updates(params, updates)

    # 5. Update privacy accounting
    state.accountant.step_poisson(
      noise_multiplier=noise_multiplier,
      sample_rate=sample_rate,
      num_steps=1,
    )

    # 6. Compute loss for logging (no gradients)
    with torch.no_grad():
      loss_value = loss_fn(params, batch).item()

    # 7. Create new state
    new_state = DPOptimizerState(
      opt_state=new_opt_state,
      accountant=state.accountant,  # Updated in-place
      noise_generator=state.noise_generator,
      step=state.step + 1,
    )

    return new_params, new_state, loss_value

  # 5. Privacy query function
  def get_privacy_spent(state: DPOptimizerState, delta: Optional[float] = None) -> tuple[float, float]:
    """Get current privacy budget spent.

    Args:
        state: Current DP optimizer state
        delta: Target delta (uses target_delta if None)

    Returns:
        (epsilon, delta)
    """
    if delta is None:
      delta = target_delta

    epsilon = state.accountant.get_epsilon(target_delta=delta)
    return epsilon, delta

  return init_fn, step_fn, get_privacy_spent
```

**Tests** (`tests/optimizers/test_dp_optimizer.py`):

```python
import torch
from opaque.optimizers import make_dp_optimizer


def test_make_dp_optimizer_sgd():
  """Test DP-SGD optimizer creation."""
  init_fn, step_fn, get_privacy = make_dp_optimizer(
    optimizer="sgd",
    learning_rate=0.1,
    l2_clip_norm=1.0,
    noise_multiplier=1.0,
    sample_rate=0.01,
  )

  # Initialize
  params = {'weight': torch.randn(10, 5)}
  state = init_fn(params)

  assert state.step == 0
  assert state.opt_state is not None


def test_dp_optimizer_single_step():
  """Test single DP-SGD step."""
  init_fn, step_fn, get_privacy = make_dp_optimizer(
    optimizer="sgd",
    learning_rate=0.1,
    l2_clip_norm=1.0,
    noise_multiplier=1.0,
    sample_rate=0.01,
  )

  # Simple linear regression
  params = torch.randn(5, requires_grad=True)
  state = init_fn(params)

  # Loss function
  def loss_fn(w, batch):
    x, y = batch
    return 0.5 * ((x @ w - y) ** 2).mean()

  # Batch data
  batch = (torch.randn(32, 5), torch.randn(32))

  # Step
  new_params, new_state, loss = step_fn(params, state, loss_fn, batch)

  assert new_state.step == 1
  assert not torch.equal(new_params, params)
  assert isinstance(loss, float)


def test_dp_optimizer_privacy_tracking():
  """Test privacy budget tracking."""
  init_fn, step_fn, get_privacy = make_dp_optimizer(
    optimizer="adam",
    learning_rate=0.001,
    l2_clip_norm=1.0,
    noise_multiplier=1.1,
    sample_rate=0.01,
  )

  params = torch.randn(10)
  state = init_fn(params)

  # Initial privacy
  eps0, delta0 = get_privacy(state)
  assert eps0 == 0.0  # No steps yet

  # Dummy loss and batch
  def loss_fn(w, batch):
    x, y = batch
    return ((x @ w - y) ** 2).mean()

  batch = (torch.randn(32, 10), torch.randn(32))

  # Take 10 steps
  for _ in range(10):
    params, state, _ = step_fn(params, state, loss_fn, batch)

  # Privacy should have increased
  eps10, delta10 = get_privacy(state)
  assert eps10 > eps0
  assert state.step == 10


def test_dp_optimizer_adam_pytree():
  """Test DP-Adam with PyTree parameters."""
  init_fn, step_fn, get_privacy = make_dp_optimizer(
    optimizer="adam",
    learning_rate=0.001,
    l2_clip_norm=1.0,
    noise_multiplier=1.0,
    sample_rate=0.01,
  )

  # Multi-layer parameters
  params = {
    'fc1': {'weight': torch.randn(64, 10), 'bias': torch.randn(64)},
    'fc2': {'weight': torch.randn(1, 64), 'bias': torch.randn(1)},
  }

  state = init_fn(params)

  # MLP loss
  def loss_fn(p, batch):
    x, y = batch
    h = torch.relu(x @ p['fc1']['weight'].T + p['fc1']['bias'])
    out = h @ p['fc2']['weight'].T + p['fc2']['bias']
    return ((out.squeeze() - y) ** 2).mean()

  batch = (torch.randn(32, 10), torch.randn(32))

  # Step
  new_params, new_state, loss = step_fn(params, state, loss_fn, batch)

  assert set(new_params.keys()) == set(params.keys())
  assert not torch.equal(new_params['fc1']['weight'], params['fc1']['weight'])
```

---

## Week 2: Training Utilities & End-to-End Workflows

### Days 1-2: Training State & Helper Functions

**File**: `src/opaque/training/__init__.py`, `src/opaque/training/dp_sgd.py`

**Goal**: Create high-level utilities for DP-SGD training loops

**Implementation**:

```python
"""Training utilities for DP-SGD."""

from typing import Any, Callable, NamedTuple, Optional
import torch
from opaque.optimizers import make_dp_optimizer, DPOptimizerState


class TrainState(NamedTuple):
  """Complete training state for DP-SGD.

  Attributes:
      params: Model parameters (PyTree)
      opt_state: DP optimizer state
      epoch: Current epoch number
  """
  params: Any
  opt_state: DPOptimizerState
  epoch: int


def create_train_state(
  params: Any,
  optimizer_config: dict,
) -> TrainState:
  """Initialize training state for DP-SGD.

  Args:
      params: Initial model parameters (PyTree)
      optimizer_config: Configuration dict for make_dp_optimizer()
          Keys: optimizer, learning_rate, l2_clip_norm, noise_multiplier, etc.

  Returns:
      TrainState with initialized optimizer

  Example:
      >>> params = {'weight': torch.randn(10, 5), 'bias': torch.randn(10)}
      >>> config = {
      ...     'optimizer': 'adam',
      ...     'learning_rate': 0.001,
      ...     'l2_clip_norm': 1.0,
      ...     'noise_multiplier': 1.1,
      ...     'sample_rate': 0.01,
      ... }
      >>> state = create_train_state(params, config)
  """
  init_fn, _, _ = make_dp_optimizer(**optimizer_config)
  opt_state = init_fn(params)

  return TrainState(
    params=params,
    opt_state=opt_state,
    epoch=0,
  )


def train_epoch(
  state: TrainState,
  dataloader: Any,
  loss_fn: Callable,
  step_fn: Callable,
) -> tuple[TrainState, float]:
  """Train for one epoch with DP-SGD.

  Args:
      state: Current training state
      dataloader: Iterable of batches
      loss_fn: Loss function with signature loss_fn(params, batch) -> scalar
      step_fn: DP-SGD step function from make_dp_optimizer()

  Returns:
      (new_state, avg_loss)

  Example:
      >>> _, step_fn, _ = make_dp_optimizer(...)
      >>> state = create_train_state(params, config)
      >>>
      >>> for epoch in range(n_epochs):
      ...     state, loss = train_epoch(state, dataloader, loss_fn, step_fn)
      ...     print(f"Epoch {epoch}: loss={loss:.4f}")
  """
  total_loss = 0.0
  n_batches = 0

  params = state.params
  opt_state = state.opt_state

  for batch in dataloader:
    params, opt_state, loss = step_fn(params, opt_state, loss_fn, batch)
    total_loss += loss
    n_batches += 1

  avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

  new_state = TrainState(
    params=params,
    opt_state=opt_state,
    epoch=state.epoch + 1,
  )

  return new_state, avg_loss


def dp_sgd_train(
  params: Any,
  dataloader: Any,
  loss_fn: Callable,
  optimizer_config: dict,
  n_epochs: int,
  log_every: int = 1,
  target_delta: float = 1e-5,
) -> tuple[Any, list[float], list[float]]:
  """Complete DP-SGD training loop.

  Args:
      params: Initial parameters (PyTree)
      dataloader: Training data loader
      loss_fn: Loss function
      optimizer_config: Configuration for make_dp_optimizer()
      n_epochs: Number of training epochs
      log_every: Print progress every N epochs
      target_delta: Target delta for privacy reporting

  Returns:
      (final_params, losses, epsilons)

  Example:
      >>> params = initialize_model()
      >>> config = {
      ...     'optimizer': 'adam',
      ...     'learning_rate': 0.001,
      ...     'l2_clip_norm': 1.0,
      ...     'noise_multiplier': 1.1,
      ...     'sample_rate': batch_size / len(dataset),
      ... }
      >>>
      >>> final_params, losses, epsilons = dp_sgd_train(
      ...     params, dataloader, loss_fn, config, n_epochs=20
      ... )
  """
  # Initialize
  init_fn, step_fn, get_privacy = make_dp_optimizer(**optimizer_config)
  state = create_train_state(params, optimizer_config)

  losses = []
  epsilons = []

  # Training loop
  for epoch in range(n_epochs):
    state, loss = train_epoch(state, dataloader, loss_fn, step_fn)
    epsilon, delta = get_privacy(state.opt_state, delta=target_delta)

    losses.append(loss)
    epsilons.append(epsilon)

    if (epoch + 1) % log_every == 0:
      print(f"Epoch {epoch + 1}/{n_epochs}: "
            f"loss={loss:.4f}, ε={epsilon:.2f}, δ={delta:.2e}")

  return state.params, losses, epsilons
```

**Tests** (`tests/training/test_dp_sgd.py`):

```python
def test_create_train_state():
  """Test training state initialization."""
  params = torch.randn(10, 5)
  config = {
    'optimizer': 'sgd',
    'learning_rate': 0.1,
    'l2_clip_norm': 1.0,
    'noise_multiplier': 1.0,
    'sample_rate': 0.01,
  }

  state = create_train_state(params, config)

  assert state.epoch == 0
  assert state.opt_state.step == 0


def test_train_epoch():
  """Test single epoch training."""
  # Setup
  params = torch.randn(5)
  config = {
    'optimizer': 'sgd',
    'learning_rate': 0.1,
    'l2_clip_norm': 1.0,
    'noise_multiplier': 1.0,
    'sample_rate': 0.01,
  }

  _, step_fn, _ = make_dp_optimizer(**config)
  state = create_train_state(params, config)

  # Simple dataloader
  def loss_fn(w, batch):
    x, y = batch
    return ((x @ w - y) ** 2).mean()

  batches = [
    (torch.randn(32, 5), torch.randn(32))
    for _ in range(10)
  ]

  # Train one epoch
  new_state, avg_loss = train_epoch(state, batches, loss_fn, step_fn)

  assert new_state.epoch == 1
  assert new_state.opt_state.step == 10
  assert isinstance(avg_loss, float)


def test_dp_sgd_train_full():
  """Test complete DP-SGD training."""
  # Linear regression problem
  torch.manual_seed(42)
  n_samples, n_features = 1000, 10
  X = torch.randn(n_samples, n_features)
  true_w = torch.randn(n_features)
  y = X @ true_w + 0.1 * torch.randn(n_samples)

  # Create batches
  batch_size = 32
  batches = []
  for i in range(0, n_samples, batch_size):
    X_batch = X[i:i + batch_size]
    y_batch = y[i:i + batch_size]
    batches.append((X_batch, y_batch))

  # DP-SGD config
  config = {
    'optimizer': 'sgd',
    'learning_rate': 0.01,
    'l2_clip_norm': 1.0,
    'noise_multiplier': 1.0,
    'sample_rate': batch_size / n_samples,
  }

  # Loss function
  def loss_fn(w, batch):
    x, y = batch
    return ((x @ w - y) ** 2).mean()

  # Train
  params = torch.randn(n_features)
  final_params, losses, epsilons = dp_sgd_train(
    params, batches, loss_fn, config, n_epochs=10, log_every=5
  )

  # Verify training happened
  assert len(losses) == 10
  assert len(epsilons) == 10
  assert epsilons[-1] > 0  # Privacy spent
  assert losses[-1] < losses[0]  # Loss decreased
```

### Days 3-5: Tutorial 03 Notebook

**File**: `docs/tutorials/03_complete_dp_sgd_training.ipynb`

**Goal**: Complete tutorial demonstrating full DP-SGD workflow

**Outline**:

1. **Introduction** - What we'll build
2. **Setup** - Data, model, configuration
3. **DP-SGD Training** - Using our optimizer API
4. **Non-Private Baseline** - For comparison
5. **Privacy-Utility Tradeoff** - Vary epsilon, compare
6. **Visualization** - Loss curves, privacy consumption
7. **Summary** - Key takeaways

**Key code examples**:

```python
# Part 1: Setup
from opaque import dp_sgd_train, calibrate_noise_multiplier

# Generate data
X, y = generate_binary_classification_data(n=10000, d=20)

# Part 2: Calibrate for target privacy
target_epsilon = 3.0
target_delta = 1e-5
batch_size = 32
n_epochs = 20
sample_rate = batch_size / len(X)
total_steps = n_epochs * (len(X) // batch_size)

noise_multiplier = calibrate_noise_multiplier(
  target_epsilon=target_epsilon,
  target_delta=target_delta,
  sample_rate=sample_rate,
  num_steps=total_steps,
)

# Part 3: Train with DP-SGD
config = {
  'optimizer': 'adam',
  'learning_rate': 0.001,
  'l2_clip_norm': 1.0,
  'noise_multiplier': noise_multiplier,
  'sample_rate': sample_rate,
}

final_params, losses, epsilons = dp_sgd_train(
  params=initialize_model(),
  dataloader=create_dataloader(X, y, batch_size),
  loss_fn=binary_cross_entropy,
  optimizer_config=config,
  n_epochs=n_epochs,
)

# Part 4: Visualize
plt.plot(losses, label='DP-SGD')
plt.plot(losses_baseline, label='Non-private')
plt.legend()
```

---

## Week 3: JAX Validation & Advanced Features

### Days 1-2: JAX-Privacy Numerical Validation

**File**: `tests/jax_validation/test_dp_sgd.py`

**Goal**: Validate complete DP-SGD matches JAX-Privacy

**Challenge**: Match training dynamics, not just individual components

**Test Strategy**:

1. Same random seed for initialization
2. Same data (convert JAX ↔ PyTorch)
3. Same optimizer hyperparameters
4. Compare parameter values after N steps
5. Tolerance: atol=1e-4 (slightly looser due to different PRNG)

```python
@pytest.mark.jax_validation
def test_dp_sgd_matches_jax_privacy():
  """Verify DP-SGD training matches JAX-Privacy."""
  # Setup problem
  torch.manual_seed(42)
  n_samples, n_features = 500, 10
  X_torch = torch.randn(n_samples, n_features)
  y_torch = torch.randn(n_samples)

  # Convert to JAX
  X_jax = jnp.array(X_torch.numpy())
  y_jax = jnp.array(y_torch.numpy())

  # Initialize parameters (same seed)
  w_init_torch = torch.randn(n_features, generator=torch.Generator().manual_seed(123))
  w_init_jax = jnp.array(w_init_torch.numpy())

  # Config
  lr = 0.01
  clip_norm = 1.0
  noise_mult = 1.0
  batch_size = 32
  n_steps = 10

  # Opaque training
  # ... (implementation)

  # JAX-Privacy training
  # ... (implementation)

  # Compare final parameters
  assert torch.allclose(
    torch.from_numpy(np.array(w_final_jax)),
    w_final_torch,
    atol=1e-4
  )
```

### Days 3-4: Advanced Optimizer Features

**Goal**: Support additional TorchOpt features

**Features to add**:

1. **Learning rate schedules**
   ```python
   # Exponential decay
   schedule = torchopt.schedule.exponential_decay(
       init_value=0.1,
       decay_rate=0.9,
       decay_steps=100,
   )

   optimizer = torchopt.sgd(lr=schedule)
   ```

2. **Gradient clipping variants**
  - Global norm clipping (already have)
  - Per-parameter clipping (for testing)

3. **Adam variants**
  - AdamW (weight decay)
  - RAdam, Lookahead (if needed)

**Implementation**:

Add optional scheduler support to `make_dp_optimizer()`:

```python
def make_dp_optimizer(
  ...,
  lr_schedule: Optional[Callable] = None,
  **optimizer_kwargs,
):
  """..."""
  if lr_schedule is not None:
    # Use scheduled learning rate
    base_optimizer = optimizer_map[optimizer](lr=lr_schedule, **optimizer_kwargs)
  else:
    # Constant learning rate
    base_optimizer = optimizer_map[optimizer](lr=learning_rate, **optimizer_kwargs)
  # ...
```

### Day 5: Documentation & Polish

**Tasks**:

1. Complete all docstrings (Google style)
2. Update main `README.md` with DP-SGD example
3. Update `CLAUDE.md` with Stage 3 completion
4. Update `roadmap.md` to mark Stage 3 complete
5. Code formatting and linting

**Module exports** (`src/opaque/__init__.py`):

```python
# Optimizers
from opaque.optimizers import make_dp_optimizer, DPOptimizerState

# Training utilities
from opaque.training import (
  create_train_state,
  train_epoch,
  dp_sgd_train,
  TrainState,
)

__all__ = [
  # ... existing
  'make_dp_optimizer',
  'DPOptimizerState',
  'create_train_state',
  'train_epoch',
  'dp_sgd_train',
  'TrainState',
]
```

---

## Week 4 (Optional): Advanced Topics & Microbatching

### Optional: Microbatching Implementation

**Goal**: Memory-efficient gradient accumulation for large batches

**Why microbatching?**

- DP-SGD benefits from large logical batch sizes (better privacy amplification)
- Large batches don't fit in GPU memory
- Solution: Process in smaller microbatches, accumulate gradients

**Implementation** (`src/opaque/training/microbatching.py`):

```python
def microbatched_step_fn(
  params,
  state: DPOptimizerState,
  loss_fn: Callable,
  batch: Any,
  microbatch_size: int,
) -> tuple[Any, DPOptimizerState, float]:
  """DP-SGD step with microbatching.

  Splits large batch into microbatches for memory efficiency.
  Accumulates clipped gradients across microbatches.

  Args:
      params: Current parameters
      state: DP optimizer state
      loss_fn: Loss function
      batch: Full batch (will be split)
      microbatch_size: Size of each microbatch

  Returns:
      (new_params, new_state, loss_value)
  """
  # Split batch into microbatches
  microbatches = split_batch(batch, microbatch_size)

  # Accumulate clipped gradients
  total_clipped_grads = None
  total_loss = 0.0

  for microbatch in microbatches:
    # Compute clipped gradients for this microbatch
    clipped_grads = clipped_grad_fn(params, microbatch)

    # Accumulate
    if total_clipped_grads is None:
      total_clipped_grads = clipped_grads
    else:
      total_clipped_grads = tree_map(
        lambda a, b: a + b,
        total_clipped_grads,
        clipped_grads
      )

    # Track loss
    with torch.no_grad():
      total_loss += loss_fn(params, microbatch).item()

  # Add noise to accumulated gradients
  noisy_grads = add_gaussian_noise(total_clipped_grads, stddev, generator)

  # Apply update (same as before)
  # ...
```

**Note**: This is optional for Stage 3. Can be deferred to Stage 4 if time is limited.

---

## Success Criteria

### Functional Requirements

1. ✅ `make_dp_optimizer()` creates working DP-SGD optimizer from TorchOpt
2. ✅ Supports SGD, Adam, AdamW optimizers
3. ✅ Integrates clipping, noise, and accounting seamlessly
4. ✅ `dp_sgd_train()` provides complete training loop
5. ✅ Privacy budget tracked accurately throughout training

### Validation Requirements

1. ✅ Numerical equivalence with JAX-Privacy DP-SGD (atol=1e-4)
2. ✅ Training converges on toy problems (linear regression, binary classification)
3. ✅ Privacy accounting matches expected values
4. ✅ Works with PyTree parameters (nested dicts)

### Code Quality

1. ✅ >85% code coverage
2. ✅ Google-style docstrings with examples
3. ✅ Passes Ruff formatting/linting
4. ✅ Tutorial 03 runs end-to-end without errors

### User Experience

1. ✅ Simple API - 3-5 lines to set up DP-SGD
2. ✅ Clear error messages
3. ✅ Comprehensive tutorial with visualizations
4. ✅ Documentation includes migration guide from Opacus

---

## Design Decisions

### 1. TorchOpt vs. Custom Implementation

**Decision**: Use TorchOpt for base optimizers

**Rationale**:

- JAX-like functional API (similar to JAX-Privacy's Optax)
- Well-tested, efficient implementations
- Supports PyTree parameters natively
- Focus our effort on DP integration, not optimizer math
- Easy to swap if needed (clean abstraction)

**Alternative considered**: Implement optimizers from scratch

- ❌ More work, error-prone
- ❌ Reinventing the wheel
- ✅ Would give more control (but not needed for Stage 3)

### 2. Wrapper API vs. Monolithic

**Decision**: Three-function API (init, step, get_privacy)

**Rationale**:

- Follows functional programming paradigm
- Clear separation of concerns
- Easy to test each function independently
- Matches JAX-Privacy pattern
- Flexible - users can customize workflow

**Alternative considered**: Single `DPOptimizer` class

- ✅ More familiar to PyTorch users
- ❌ Stateful (conflicts with functional design)
- ❌ Harder to customize

### 3. Training State Management

**Decision**: Explicit `TrainState` NamedTuple

**Rationale**:

- Immutable state (functional paradigm)
- Clear what's being tracked
- Easy to serialize/deserialize
- Type-safe with NamedTuple

### 4. Microbatching in Stage 3 or 4?

**Decision**: Optional for Stage 3, defer to Stage 4 if needed

**Rationale**:

- Core DP-SGD doesn't require microbatching
- Can train on toy problems without it
- Stage 4 focuses on large-scale training (where it's needed)
- Avoids scope creep in Stage 3

---

## Reference: JAX-Privacy Patterns

**JAX-Privacy DP-SGD pattern**:

```python
# JAX-Privacy (from documentation/examples)
import optax
from jax_privacy.experimental.clipping import clipped_grad

# 1. Define loss and create clipped grad
clipped_grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)

# 2. Create optimizer with noise
optimizer = optax.chain(
  optax.scale_by_adam(),
  optax.add_noise(noise_multiplier=1.0),
  optax.scale(-learning_rate),
)
opt_state = optimizer.init(params)


# 3. Training step
def step(params, opt_state, batch):
  grads = clipped_grad_fn(params, batch)
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return params, opt_state
```

**Our Opaque pattern** (parallel structure):

```python
# Opaque (this stage)
from opaque import make_dp_optimizer

# 1. Create DP optimizer (combines clipping + noise)
init_fn, step_fn, get_privacy = make_dp_optimizer(
  optimizer='adam',
  l2_clip_norm=1.0,
  noise_multiplier=1.0,
  learning_rate=0.001,
)

# 2. Initialize
state = init_fn(params)

# 3. Training step
params, state, loss = step_fn(params, state, loss_fn, batch)
epsilon, delta = get_privacy(state)
```

---

## Dependencies Update

**Add to `pyproject.toml`**:

```toml
[project]
dependencies = [
  "torch>=2.0.0",
  "dp-accounting>=0.4.0",
  "optree>=0.11.0",
  "torchopt>=0.7.3", # NEW: Functional optimizers
]
```

---

## What Comes Next

**Stage 4: High-Level API & LoRA Integration** (2 weeks)

- `make_private()` one-line wrapper
- Integration with HuggingFace PEFT library
- Automatic LoRA parameter detection
- Real LLM fine-tuning examples

**Stage 5: Polish & Documentation** (2-3 weeks)

- Complete tutorial series
- Performance optimization
- Benchmark suite
- PyPI publication

See [Project Roadmap](roadmap.md) for details.
