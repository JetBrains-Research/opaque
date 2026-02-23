"""Privacy auditing: verify DP guarantees empirically.

Trains a simple model with DP-SGD, then audits the training run using
membership inference to estimate the empirical privacy loss (epsilon).

Usage:
    # Quick audit (~2 minutes on CPU)
    python examples/audit_model.py

    # More canaries for tighter bounds
    python examples/audit_model.py --num-canaries 2000

    # Compare with specific theoretical epsilon
    python examples/audit_model.py --target-epsilon 3.0 --target-delta 1e-5

The audited epsilon should be below the theoretical epsilon. If it exceeds
the theoretical bound, there is likely a bug in the DP implementation.
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchopt
from torch.utils.data import DataLoader, TensorDataset

import opaque.accounting as acc
import opaque.auditing as auditing
from opaque import clipped_grad, gaussian_noise
from opaque.random import key, split
from opaque.sampling import PoissonSampler


def make_dataset(n: int, d: int, seed: int = 0):
    """Generate a synthetic classification dataset."""
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=gen)
    w_true = torch.randn(d, generator=gen)
    y = (X @ w_true > 0).long()
    return TensorDataset(X, y)


class SimpleModel(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def run_audit(args):
    rng_key = key(args.seed)
    key_audit, key_samp, key_noise = split(rng_key, num=3)

    # Dataset
    dataset = make_dataset(args.num_samples, args.dim, seed=args.seed)
    n = len(dataset)
    sample_rate = args.batch_size / n

    # Privacy calibration
    total_steps = args.num_steps
    result = acc.calibrate(
        acc.epsilon_budget(args.target_epsilon, delta=args.target_delta),
        lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=sample_rate) * total_steps,
        param_min=0.1,
        param_max=100.0,
    )
    noise_multiplier = result.param
    theoretical_eps = result.value

    print("=" * 60)
    print("Privacy Auditing")
    print("=" * 60)
    print(f"Dataset:            {n} samples, {args.dim}D")
    print(f"Training steps:     {total_steps}")
    print(f"Noise multiplier:   {noise_multiplier:.4f}")
    print(f"Theoretical epsilon: {theoretical_eps:.4f}")
    print(f"Target delta:       {args.target_delta}")
    print(f"Canaries:           {args.num_canaries}")
    print()

    # Auditing setup: designate canaries and flip coins
    experiment = auditing.setup(dataset, num_canaries=args.num_canaries, key=key_audit)

    # Train on the subset (canaries randomly included/excluded)
    train_data = experiment.subset(dataset)
    sampler = PoissonSampler(
        train_data,
        sample_rate=sample_rate,
        num_epochs=total_steps,
        key=key_samp,
    )
    train_loader = DataLoader(train_data, batch_sampler=sampler)

    # Model setup
    model = SimpleModel(args.dim)
    from opaque import make_functional

    fmodel, params = make_functional(model)

    def loss_fn(params, x, y):
        logits = fmodel(params, x.unsqueeze(0))
        return F.cross_entropy(logits.squeeze(0), y.unsqueeze(0))

    grad_fn, clip_state = clipped_grad(
        loss_fn,
        l2_clip_norm=args.clip_norm,
        argnums=0,
        batch_argnums=(1, 2),
    )
    noise_fn, noise_state = gaussian_noise(
        stddev=noise_multiplier * clip_state.sensitivity(),
        key=key_noise,
    )
    optimizer = torchopt.adam(lr=args.lr)
    opt_state = optimizer.init(params)

    # Training loop
    print("Training...", end=" ", flush=True)
    t0 = time.time()
    step = 0
    for batch in train_loader:
        x, y = batch[0], batch[1]
        if len(x) == 0:
            continue
        grads, clip_state = grad_fn(params, x, y, state=clip_state)
        noisy_grads, noise_state = noise_fn(grads, noise_state)
        updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
        params = torchopt.apply_updates(params, updates)
        step += 1
    elapsed = time.time() - t0
    print(f"done ({step} steps, {elapsed:.1f}s)")

    # Evaluate audit
    print("Scoring canaries...", end=" ", flush=True)
    audit_result = auditing.evaluate(experiment, loss_fn, params, dataset)
    print("done")

    # Print results
    print()
    print("=" * 60)
    print("Audit Results")
    print("=" * 60)

    empirical_eps = audit_result.epsilon_at(delta=args.target_delta)
    print(f"Theoretical epsilon:  {theoretical_eps:.4f}")
    print(f"Empirical epsilon:    {empirical_eps:.4f}")

    gap = theoretical_eps - empirical_eps
    if gap >= 0:
        print(f"Gap:                  {gap:.4f} (healthy: empirical < theoretical)")
    else:
        print(f"Gap:                  {gap:.4f} (WARNING: empirical > theoretical)")

    print()
    print("Attack metrics:")
    print(f"  AUC:                {audit_result.auc():.4f}")
    print(f"  beta(alpha=0.01):   {audit_result.beta_at(alpha=0.01):.4f}")
    print(f"  beta(alpha=0.10):   {audit_result.beta_at(alpha=0.10):.4f}")

    if hasattr(audit_result, "summary"):
        print()
        print(audit_result.summary())


def main():
    parser = argparse.ArgumentParser(description="Privacy auditing example")
    # Dataset
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=20)
    # Training
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    # Privacy
    parser.add_argument("--target-epsilon", type=float, default=3.0)
    parser.add_argument("--target-delta", type=float, default=1e-5)
    # Auditing
    parser.add_argument("--num-canaries", type=int, default=1000)
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
