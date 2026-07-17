"""ZenML step/pipeline settings for opaque training, built with ``jb-mlops``.

This is the opaque counterpart to ``next-edit-pipeline``'s
``jetbrains/nes/zenml/base.py`` + ``pipelines/training/settings.py``. It uses
the shared ``jetbrains.mlops.zenml.get_step_settings`` helper (the same one NES
uses) to produce the Kubernetes orchestrator + Docker settings for a GPU
training pod, so opaque runs on the non-TRACE GKE stack exactly the way NES
training does.

Secrets are injected into the pod as env vars via native Kubernetes
``secretKeyRef`` entries; the kubelet resolves them at pod start, so a
credential never lands in the pod spec or the ZenML run metadata. ``HF_TOKEN``
comes from the shared ``ai-for-code`` Secret (key ``hf_token``, reused from NES).
The W&B key comes from a dedicated ``opaque-wandb`` Secret because the shared
ai-for-code W&B key is not a ``federated-compute`` team member and so cannot
write to the ``opaque-lora-xs`` project; ``opaque-wandb`` holds a team-member
key. (A ZenML ``{{secret.key}}`` reference is *not* used because ZenML 0.94 does
not substitute such references placed in raw Kubernetes pod env.)

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

# Native k8s Secrets (in the orchestrator namespace) backing the pod creds via
# ``secretKeyRef``. ``ai-for-code`` (shared, from NES) provides HF_TOKEN;
# ``opaque-wandb`` (dedicated) provides a WANDB_API_KEY that belongs to a
# ``federated-compute`` team member. Both names are env-overridable.
DEFAULT_ZENML_SECRET = os.environ.get("OPAQUE_ZENML_SECRET", "ai-for-code")
DEFAULT_WANDB_SECRET = os.environ.get("OPAQUE_WANDB_SECRET", "opaque-wandb")
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


def secret_env(env_name: str, *, key: str | None = None, secret: str = DEFAULT_ZENML_SECRET) -> dict[str, Any]:
    """A pod env var sourced from the native k8s ``Secret`` via ``secretKeyRef``.

    ``key`` is the key *inside* the Secret and defaults to ``env_name``. The
    cluster Secret stores ``WANDB_API_KEY`` under a matching key but the HF token
    under lowercase ``hf_token`` (see ``hf_envs``). The kubelet injects the value
    at pod start, so it is never written into the pod spec or run metadata.
    """
    return {"name": env_name, "valueFrom": {"secretKeyRef": {"name": secret, "key": key or env_name}}}


def wandb_envs() -> list[dict[str, Any]]:
    envs: list[dict[str, Any]] = [
        {"name": "WANDB_BASE_URL", "value": os.environ.get("WANDB_BASE_URL", DEFAULT_WANDB_BASE_URL)},
        {"name": "WANDB_ENTITY", "value": os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY)},
        {"name": "WANDB_PROJECT", "value": os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT)},
        secret_env("WANDB_API_KEY", secret=DEFAULT_WANDB_SECRET),
    ]
    # Propagate an explicit WANDB_MODE (e.g. ``offline``) from the submitting env
    # so a smoke can validate the whole pipeline without W&B write access to the
    # target project. Left unset, the trainer picks online when a key is present.
    if mode := os.environ.get("WANDB_MODE"):
        envs.append({"name": "WANDB_MODE", "value": mode})
    return envs


def hf_envs() -> list[dict[str, Any]]:
    # The shared ``ai-for-code`` k8s Secret stores the HF token under the
    # lowercase key ``hf_token`` (the WANDB key matches its env name).
    return [secret_env("HF_TOKEN", key="hf_token")]


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
        # These non-TRACE GKE stacks enforce ``runAsNonRoot``. The kubelet can
        # only verify that from a *numeric* UID, so a bare ``USER nonroot`` image
        # is rejected with ``CreateContainerConfigError: container has
        # runAsNonRoot and image has non-numeric user (nonroot)``. Pin the pod
        # securityContext to UID/GID 1000 (the image's ``nonroot`` user) so the
        # check passes with any image. ``fsGroup`` makes the mounted /scratch +
        # /tmp PVCs group-writable by that user (they mount root-owned, so a
        # non-root process otherwise can't write to them). pod_spec.security_context
        # starts unset, so this dict is applied verbatim (clean camelCase keys).
        additional_pod_spec_args={
            "security_context": {
                "runAsUser": 1000,
                "runAsGroup": 1000,
                "runAsNonRoot": True,
                "fsGroup": 1000,
            }
        },
    )

    # Submit asynchronously, exactly like NES (`pusk/launch.py` forces
    # ``synchronous=False`` on every run): the client returns as soon as the
    # orchestrator Job is created instead of blocking for the whole run, and pod
    # startup/retries are left to the Kubernetes Job. The run is then monitored
    # via the dashboard / ZenML API. (Non-root enforcement is handled by the pod
    # securityContext set on ``pod_configuration`` above.)
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
