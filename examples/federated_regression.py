"""Differentially private federated linear regression, driven round by round.

The federated twin of a central DP-SGD loop: a cohort of clients replaces the
batch axis, IFED runs each round, and everything after the gradient — noise,
optimizer, accounting — is unchanged central Opaque.

Run it as ``uv run python examples/federated_regression.py [target]``, where
target is "local" (two agents in this process, over the rows below), "prod",
"stgn" or a driver base URL.
"""

import argparse
import os

import ifed
import torch
import torchopt
from torch import nn

import opaque.federated as fed
from opaque.dpsgd.noise import gaussian_noise
from opaque.optimizers import sgd
from opaque.random import key

ROWS = [
    {"sepal_length": 5.1, "sepal_width": 3.5},
    {"sepal_length": 4.9, "sepal_width": 3.0},
    {"sepal_length": 4.7, "sepal_width": 3.2},
    {"sepal_length": 4.6, "sepal_width": 3.1},
    {"sepal_length": 5.0, "sepal_width": 3.6},
    {"sepal_length": 5.4, "sepal_width": 3.9},
    {"sepal_length": 6.4, "sepal_width": 3.2},
    {"sepal_length": 6.9, "sepal_width": 3.1},
    {"sepal_length": 5.5, "sepal_width": 2.3},
    {"sepal_length": 6.5, "sepal_width": 2.8},
]


class Iris(ifed.Dataset):
    sepal_length = ifed.Float()
    sepal_width = ifed.Float()


class Regression(nn.Module):
    """Predict sepal width from sepal length."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features).squeeze(-1)


def held_out_loss(params: dict) -> float:
    features = torch.tensor([[row["sepal_length"]] for row in ROWS])
    targets = torch.tensor([row["sepal_width"] for row in ROWS])
    prediction = features @ params["fc.weight"].T + params["fc.bias"]
    return float(((prediction.squeeze(-1) - targets) ** 2).mean())


def datastore(target: str, sampler: fed.MinSepSampler):
    """Where the clients are: "local" is this process, anything else is a driver."""
    if target == "local":
        # one dataset entry per agent, so listing it twice is what makes two clients
        iris = ifed.BuiltInDataset("Iris", ROWS)
        return ifed.LocalDatastore(datasets=[iris] * sampler.batch_size)
    return fed.datastore(sampler, server=target, gpu=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target", nargs="?", default=os.environ.get("IFED_SERVER", "local")
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--bands", type=int, default=1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.5)
    args = parser.parse_args()

    population = fed.population("/hive")
    sampler = fed.MinSepSampler(population, batch_size=args.clients, bands=args.bands)
    loader = fed.DataLoader(population, batch_sampler=sampler, rounds=args.rounds)

    strategy = fed.clipped_sum(clipping_norm=args.clip)
    plan = ifed.build_train(
        net=Regression(),
        source=Iris,
        target="sepal_width",
        features=["sepal_length"],
        loss=ifed.Loss.mse,
        batch_size=None,  # one client contribution = one full-batch gradient
        shuffle=False,
        strategy=strategy,
    )

    store = datastore(args.target, sampler)
    # assign_delta is a driver setting: a local run has no driver to enforce it
    driver = (
        {}
        if isinstance(store, ifed.LocalDatastore)
        else {"assign_delta": sampler.assign_delta}
    )

    with ifed.session(plan, store, **driver) as run:
        params = plan.init(plan.input_dir).params
        grad_fn, clip_state = fed.clipped_grad(run, strategy)
        noise_fn, noise_state = gaussian_noise(noise_multiplier=args.sigma, key=key(42))
        optimizer = sgd(lr=args.lr)
        opt_state = optimizer.init(params)

        for step, cohort in enumerate(loader, start=1):
            gradients, clip_state = grad_fn(params, cohort, state=clip_state)
            noised, noise_state = noise_fn(gradients, noise_state)
            updates, opt_state = optimizer.update(noised, opt_state)
            params = torchopt.apply_updates(params, updates)
            print(f"round {step:3d}  held-out mse={held_out_loss(params):.4f}")

    print("done")


if __name__ == "__main__":
    main()
