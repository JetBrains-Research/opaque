"""DP-SGD image classification on CIFAR-10 or MNIST.

Production-grade example with privacy calibration, Poisson sampling,
and evaluation. Demonstrates the full Opaque DP-SGD pipeline.

Quick run (~3 minutes on CPU with MNIST):
    python examples/train_classifier.py --dataset mnist --epochs 5

Serious training (GPU recommended):
    python examples/train_classifier.py --dataset cifar10 --epochs 20 \
        --target-epsilon 3.0 --batch-size 1024

All training satisfies (epsilon, delta)-DP. The noise multiplier is
automatically calibrated to meet the target epsilon.
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchopt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import opaque.accounting as acc
from opaque import clipped_grad, gaussian_noise
from opaque.random import key, split
from opaque.sampling import PoissonSampler


# --- Models ---


class SimpleCNN(nn.Module):
    """Small CNN for MNIST / CIFAR-10."""

    def __init__(self, in_channels: int, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(start_dim=-3)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# --- Data ---


def get_dataset(name: str, train: bool):
    if name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        return datasets.MNIST("data", train=train, download=True, transform=transform)
    elif name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )
        return datasets.CIFAR10("data", train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")


# --- Training ---


def train(args):
    device = torch.device(args.device)
    rng_key = key(args.seed)
    key_samp, key_noise = split(rng_key, num=2)

    # Data
    train_dataset = get_dataset(args.dataset, train=True)
    test_dataset = get_dataset(args.dataset, train=False)
    in_channels = 1 if args.dataset == "mnist" else 3
    n = len(train_dataset)
    sample_rate = args.batch_size / n
    steps_per_epoch = math.ceil(1 / sample_rate)
    total_steps = args.epochs * steps_per_epoch

    # Model (functional form)
    model = SimpleCNN(in_channels=in_channels).to(device)
    from opaque import make_functional

    fmodel, params = make_functional(model)

    def loss_fn(params, x, y):
        logits = fmodel(params, x.unsqueeze(0))
        return F.cross_entropy(logits.squeeze(0), y.unsqueeze(0))

    # Privacy calibration
    print(f"Dataset: {args.dataset} ({n} examples)")
    print(f"Sample rate: {sample_rate:.4f}, Steps: {total_steps}")
    print(f"Target: (epsilon={args.target_epsilon}, delta={args.target_delta})-DP")

    result = acc.calibrate(
        acc.epsilon_budget(args.target_epsilon, delta=args.target_delta),
        lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=sample_rate) * total_steps,
        param_min=0.1,
        param_max=100.0,
    )
    noise_multiplier = result.param
    achieved_eps = result.value
    print(f"Calibrated noise multiplier: {noise_multiplier:.4f}")
    print(f"Achieved epsilon: {achieved_eps:.4f}")

    # DP components
    grad_fn, clip_state = clipped_grad(
        loss_fn,
        l2_clip_norm=args.clip_norm,
        argnums=0,
        batch_argnums=(1, 2),
        microbatch_size=args.microbatch_size,
    )
    noise_fn, noise_state = gaussian_noise(
        stddev=noise_multiplier * clip_state.sensitivity(),
        key=key_noise,
    )

    # Optimizer
    optimizer = torchopt.adam(lr=args.lr)
    opt_state = optimizer.init(params)

    # Sampler
    sampler = PoissonSampler(
        train_dataset,
        sample_rate=sample_rate,
        num_epochs=total_steps,
        key=key_samp,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=sampler)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs ({total_steps} steps)...")
    step = 0
    t0 = time.time()

    for batch in train_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        if len(x) == 0:
            continue

        grads, clip_state = grad_fn(params, x, y, state=clip_state)
        noisy_grads, noise_state = noise_fn(grads, noise_state)
        updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
        params = torchopt.apply_updates(params, updates)

        step += 1
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"  Step {step}/{total_steps} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f}s")

    # Evaluation
    test_loader = DataLoader(test_dataset, batch_size=256)
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = fmodel(params, x)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += len(y)

    accuracy = correct / total
    print(f"\nTest accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Privacy guarantee: ({achieved_eps:.2f}, {args.target_delta})-DP")


def main():
    parser = argparse.ArgumentParser(description="DP-SGD image classification")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=None,
        help="Microbatch size for memory efficiency (default: full batch)",
    )
    parser.add_argument("--target-epsilon", type=float, default=3.0)
    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
