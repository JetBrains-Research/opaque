#!/usr/bin/env python
"""Submit the Rényi DP experiments to the non-TRACE ZenML GPU stack.

Mirrors how ``next-edit-pipeline`` launches training with ZenML: activate the
project + GPU stack, attach ``jb-mlops`` step settings (Docker image, H100 pod,
secret env vars), then submit the pipeline. The two arms are the exact commands
from ``docs/renyi-dp-experiment-handoff.md`` / ``.cadence/configs/renyi_dp_vs_nodp.yaml``.

Prerequisites (see deploy/zenml/README.md):
  * ``zenml login <server>`` once (or pass --login / OPAQUE_ZENML_LOGIN),
  * ``jb-mlops[zenml]`` installed locally (space-tools index; creds in env),
  * the ``opaque-train`` image built + pushed (see build_and_push.sh).

Usage:
    # cheap smoke test (tiny-gpt2, CPU) — validates wiring end to end
    python deploy/zenml/run.py smoke --no-gpu

    # the real two-arm experiment (Qwen2.5-Coder-7B, H100)
    python deploy/zenml/run.py dp
    python deploy/zenml/run.py nodp
    python deploy/zenml/run.py both      # dp + nodp back to back

    # show the resolved stack / image / GPU / argv without submitting
    python deploy/zenml/run.py dp --dry-run

Every infrastructure value is env-overridable so you don't edit this file.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# --------------------------------------------------------------------------- #
# Infrastructure config (all env-overridable). Defaults reuse the shared        #
# non-TRACE resources NES training uses.                                        #
# --------------------------------------------------------------------------- #
PROJECT = os.environ.get("OPAQUE_ZENML_PROJECT", "models-rd")
# Default to the plain non-TRACE GPU stack (Kubernetes orchestrator
# `zenml-workload-common-gpus`, GCS artifact store, GCP registry). It has no
# Slack alerter, so the training image only needs `zenml[connectors-gcp]` — no
# `slack-sdk`. `gke-ai-for-code` also works but its Slack alerter requires
# `slack-sdk` baked into the image. Project stays `models-rd` so the
# `ai-for-code` secret (WANDB/HF) resolves regardless of stack.
STACK = os.environ.get("OPAQUE_ZENML_STACK", "gke-europe-west4")
LOGIN_TARGET = os.environ.get("OPAQUE_ZENML_LOGIN", "").strip()
SCRATCH_DIR = os.environ.get("OPAQUE_SCRATCH_DIR", "/scratch")

# The exact commands from docs/renyi-dp-experiment-handoff.md (Part III/IV).
ARMS: dict[str, tuple[list[str], str]] = {
    "smoke": (
        [
            "--preset", "custom",
            "--model-name", "sshleifer/tiny-gpt2",
            "--num-train-samples", "512",
            "--num-epochs", "1",
            "--lora-method", "lora-xs",
            "--lora-xse-p-e", "0.333",
            "--optimizer", "sgd",
            "--sgd-momentum", "0.9",
        ],
        "renyi-smoke",
    ),
    "dp": (
        [
            "--preset", "qwen-coder-kstack-lora",
            "--lora-method", "lora-xs",
            "--lora-xse-p-e", "0.333",
            "--num-epochs", "1",
        ],
        "renyi-alpha-dp-eps3",
    ),
    "nodp": (
        [
            "--preset", "qwen-coder-kstack-lora",
            "--lora-method", "lora-xs",
            "--lora-xse-p-e", "0.333",
            "--num-epochs", "1",
            "--noise-multiplier", "0",
        ],
        "renyi-alpha-nodp",
    ),
}


def _login_if_requested() -> None:
    if LOGIN_TARGET:
        subprocess.run(["zenml", "login", LOGIN_TARGET], check=True)


def _resolve_cmd_args(arm: str, seed: int | None, extra: list[str]) -> tuple[list[str], str]:
    base_args, default_name = ARMS[arm]
    cmd_args = list(base_args)
    if seed is not None:
        cmd_args += ["--seed", str(seed)]
    if extra:
        cmd_args += extra
    return cmd_args, default_name


def submit(arm: str, run_name: str | None, seed: int | None, gpu: bool, extra: list[str], dry_run: bool) -> None:
    cmd_args, default_name = _resolve_cmd_args(arm, seed, extra)
    run_name = run_name or default_name

    # Imported lazily so --dry-run works without jb-mlops / zenml installed
    # (docker_images is pure-stdlib; settings pulls in jb-mlops + zenml).
    from docker_images import DockerImage, get_image_url

    if dry_run:
        image = get_image_url(DockerImage.TRAIN)
        print("DRY RUN — nothing submitted\n")
        print(f"  arm          : {arm}")
        print(f"  run_name     : {run_name}")
        print(f"  project      : {PROJECT}")
        print(f"  stack        : {STACK}")
        print(f"  image        : {image}")
        print(f"  gpu          : {'H100 x1' if gpu else 'none (cpu)'}")
        print(f"  trainer argv : examples/train_causal_lm.py {' '.join(cmd_args)}")
        return

    from settings import training_settings

    settings = training_settings(gpu=gpu)

    from zenml.client import Client
    from zenml.utils import source_utils

    # Make ``pipeline`` importable in the pod by uploading this dir as the code
    # root (mirrors NES's custom-source-root handling in pusk/launch.py).
    source_utils.set_custom_source_root(str(HERE))

    _login_if_requested()
    client = Client()
    client.set_active_project(PROJECT)
    client.activate_stack(STACK)

    from pipeline import renyi_training_pipeline

    configured = renyi_training_pipeline.with_options(
        settings=settings,
        run_name=run_name + "-{date}_{time}",
    )
    configured(cmd_args=cmd_args, run_name=run_name, scratch_dir=SCRATCH_DIR)
    print(f"submitted arm={arm} run_name={run_name} -> project={PROJECT} stack={STACK}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("smoke", "dp", "nodp", "both"):
        p = sub.add_parser(name, help=f"submit the {name} arm(s)")
        p.add_argument("--run-name", default=None, help="override the W&B / run name")
        p.add_argument("--seed", type=int, default=None, help="append --seed to the trainer")
        p.add_argument("--dry-run", action="store_true", help="print resolved config, submit nothing")
        p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra trainer flags (must be last)")
        if name == "smoke":
            p.add_argument("--no-gpu", action="store_true", help="run tiny-gpt2 on CPU (no GPU node needed)")

    args = parser.parse_args()
    gpu = not getattr(args, "no_gpu", False)

    if args.command == "both":
        submit("dp", args.run_name, args.seed, gpu, args.extra, args.dry_run)
        submit("nodp", None, args.seed, gpu, args.extra, args.dry_run)
    else:
        submit(args.command, args.run_name, args.seed, gpu, args.extra, args.dry_run)


if __name__ == "__main__":
    main()
