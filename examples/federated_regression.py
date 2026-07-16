"""Differentially private federated linear regression on Iris.

Run through IFED so the runner owns the simulator and injects its endpoint::

    ifed run --simulate examples/federated_regression_simulation.yaml \
      examples/federated_regression.py
"""

import argparse

import ifed
import torch
import torchopt

import opaque.federated as fed
from opaque.dpsgd.noise import gaussian_noise
from opaque.optimizers import sgd
from opaque.random import key


class Iris(ifed.Dataset):
    sepal_length = ifed.Float()
    sepal_width = ifed.Float()
    petal_length = ifed.Float()
    petal_width = ifed.Float()
    species = ifed.Float()


def loss_fn(
    params: dict[str, torch.Tensor], data: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Predict sepal width from sepal length."""
    prediction = data["sepal_length"].unsqueeze(1) @ params["w"] + params["b"]
    return ((prediction.squeeze(-1) - data["sepal_width"]) ** 2).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.5)
    args = parser.parse_args()

    params = {"w": torch.zeros(1, 1), "b": torch.zeros(1)}
    eval_data = {
        "sepal_length": torch.tensor(
            [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 6.4, 6.9, 5.5, 6.5]
        ),
        "sepal_width": torch.tensor([3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.2, 3.1, 2.3, 2.8]),
    }

    population = fed.population("/hive")
    sampler = fed.MinSepSampler(population, batch_size=2, bands=1)
    loader = fed.DataLoader(population, batch_sampler=sampler, rounds=args.rounds)

    with ifed.Client() as client:
        grad_fn, clip_state = fed.clipped_grad(
            loss_fn,
            client,
            clipping_norm=args.clip,
            params=params,
            data=Iris,
        )
        noise_fn, noise_state = gaussian_noise(noise_multiplier=args.sigma, key=key(42))
        optimizer = sgd(lr=args.lr)
        opt_state = optimizer.init(params)

        for step, cohort in enumerate(loader, start=1):
            gradients, clip_state = grad_fn(params, cohort, state=clip_state)
            noised, noise_state = noise_fn(gradients, noise_state)
            updates, opt_state = optimizer.update(noised, opt_state)
            params = torchopt.apply_updates(params, updates)

            value = loss_fn(params, eval_data)
            print(
                f"round {step:3d}  loss={float(value):.4f}"
                f"  w={float(params['w']):.3f}  b={float(params['b']):.3f}"
            )

    print("done")


if __name__ == "__main__":
    main()
