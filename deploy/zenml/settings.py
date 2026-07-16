"""ZenML step/pipeline settings for opaque training, built with ``jb-mlops``.

This is the opaque counterpart to ``next-edit-pipeline``'s
``jetbrains/nes/zenml/base.py`` + ``pipelines/training/settings.py``. It uses
the shared ``jetbrains.mlops.zenml.get_step_settings`` helper (the same one NES
uses) to produce the Kubernetes orchestrator + Docker settings for a GPU
training pod, so opaque runs on the non-TRACE GKE stack exactly the way NES
training does.

Secrets are injected into the pod as env vars via ZenML secret references
(``{{ai-for-code.WANDB_API_KEY}}``) — reusing the same shared ``ai-for-code``
ZenML secret NES already relies on, so no opaque-specific secret is required.

Only imported on the submitting side (laptop / CI), where ``jb-mlops`` is
installed from the ``space-tools`` index; the training image does not need it.
"""

from __future__ import annotations

import os
from typing import Any

from jetbrains.mlops.enums import GPUs
from jetbrains.mlops.zenml import MountConfiguration, PodConfiguration, get_step_settings

from docker_images import DockerImage, get_image_url

__all__ = ["training_settings", "wandb_envs", "hf_envs", "secret_env"]

# Shared ZenML secret holding WANDB_API_KEY + HF_TOKEN (created for NES; reused
# verbatim). Override with OPAQUE_ZENML_SECRET if you keep creds elsewhere.
DEFAULT_ZENML_SECRET = os.environ.get("OPAQUE_ZENML_SECRET", "ai-for-code")
DEFAULT_WANDB_BASE_URL = "https://jetbrains.wandb.io"
DEFAULT_WANDB_ENTITY = "federated-compute"
DEFAULT_WANDB_PROJECT = "opaque-lora-xs"

# H100 80GB matches the Cadence renyi preset (gpu_type: h100) and is the largest
# GPU the shared jb-mlops GPUs enum / GKE cluster exposes.
DEFAULT_MEMORY_GB = float(os.environ.get("OPAQUE_MEMORY_GB", "160"))
DEFAULT_MEMORY_LIMIT_GB = float(os.environ.get("OPAQUE_MEMORY_LIMIT_GB", "200"))
DEFAULT_CPU_COUNT = float(os.environ.get("OPAQUE_CPU_COUNT", "20"))
DEFAULT_SCRATCH_GB = float(os.environ.get("OPAQUE_SCRATCH_GB", "200"))
DEFAULT_TMP_GB = float(os.environ.get("OPAQUE_TMP_GB", "100"))
SCRATCH_DIR = os.environ.get("OPAQUE_SCRATCH_DIR", "/scratch")


def secret_env(env_name: str, *, field: str | None = None, secret: str = DEFAULT_ZENML_SECRET) -> dict[str, Any]:
    """A pod env var sourced from a ZenML secret (``{{secret.field}}``)."""
    return {"name": env_name, "value": "{{" + f"{secret}.{field or env_name}" + "}}"}


def wandb_envs() -> list[dict[str, Any]]:
    return [
        {"name": "WANDB_BASE_URL", "value": os.environ.get("WANDB_BASE_URL", DEFAULT_WANDB_BASE_URL)},
        {"name": "WANDB_ENTITY", "value": os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY)},
        {"name": "WANDB_PROJECT", "value": os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT)},
        secret_env("WANDB_API_KEY"),
    ]


def hf_envs() -> list[dict[str, Any]]:
    return [secret_env("HF_TOKEN")]


def training_settings(
    *,
    gpu: bool = True,
    docker_image_tag: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Kubernetes + Docker settings for one opaque training pod.

    Args:
        gpu: request an H100 (``False`` runs CPU-only, e.g. the tiny-gpt2 smoke).
        docker_image_tag: pin a specific image tag (defaults to ``OPAQUE_DOCKER_TAG``).
        **kwargs: forwarded to ``get_step_settings`` (e.g. ``slack_channel_id``).
    """
    extra_envs = [
        *wandb_envs(),
        *hf_envs(),
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        # DP-SGD peaks memory at the full-vocab logits copy; expandable_segments
        # reclaims fragmented reserved memory that otherwise OOMs mid-run.
        {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
    ]

    pod_configuration = PodConfiguration(
        memory_gb=DEFAULT_MEMORY_GB,
        memory_limit_gb=DEFAULT_MEMORY_LIMIT_GB,
        cpu_count=DEFAULT_CPU_COUNT,
        ephemeral_storage_gb=5,
        gpu=GPUs.H100 if gpu else None,
        gpu_count=1 if gpu else None,
        extra_envs=extra_envs,
        storage_configuration=[
            MountConfiguration(mount_path=SCRATCH_DIR, size_gb=DEFAULT_SCRATCH_GB),
            MountConfiguration(mount_path="/tmp", size_gb=DEFAULT_TMP_GB),
        ],
    )

    # Submit asynchronously, exactly like NES (`pusk/launch.py` forces
    # ``synchronous=False`` on every run): the client returns as soon as the
    # orchestrator Job is created instead of blocking for the whole run, and pod
    # startup/retries are left to the Kubernetes Job. The run is then monitored
    # via the dashboard / ZenML API. (The training image must run as non-root, or
    # the pod fails with ``CreateContainerConfigError: container has runAsNonRoot
    # and image will run as root`` — see deploy/zenml/Dockerfile.)
    orchestrator_kwargs = {"synchronous": False, **kwargs.pop("orchestrator_kwargs", {})}
    # Optional: keep finished/failed Jobs (and their pods/events) around for
    # post-mortem inspection instead of the stack's aggressive default TTL.
    if (ttl := os.environ.get("OPAQUE_JOB_TTL_SECONDS")):
        orchestrator_kwargs.setdefault("ttl_seconds_after_finished", int(ttl))

    return get_step_settings(
        docker_image=get_image_url(DockerImage.TRAIN, tag=docker_image_tag),
        pod_configuration=pod_configuration,
        orchestrator_kwargs=orchestrator_kwargs,
        **kwargs,
    )
