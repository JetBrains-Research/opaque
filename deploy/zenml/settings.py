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

# Step (trainer) pod resources. H100 80GB + 20 CPU + 160Gi RAM match the Cadence
# renyi preset (``.cadence/configs/renyi_dp_vs_nodp.yaml`` provisioning block).
# OPAQUE_GPU_TYPE selects the accelerator (one of the ``GPUs`` enum names: H100,
# A100_80GB, A100, L4). To fall back off a saturated H100 pool, set
# ``OPAQUE_GPU_TYPE=A100_80GB`` -- and lower OPAQUE_CPU_COUNT / OPAQUE_MEMORY_GB
# to fit the smaller single-A100 node shape (a2-ultragpu-1g is ~12 vCPU/170Gi,
# vs the H100 a3-highgpu-1g's ~26 vCPU/234Gi).
DEFAULT_GPU_TYPE = os.environ.get("OPAQUE_GPU_TYPE", "H100")
DEFAULT_MEMORY_GB = float(os.environ.get("OPAQUE_MEMORY_GB", "160"))
DEFAULT_MEMORY_LIMIT_GB = float(os.environ.get("OPAQUE_MEMORY_LIMIT_GB", "200"))
DEFAULT_CPU_COUNT = float(os.environ.get("OPAQUE_CPU_COUNT", "20"))
DEFAULT_SCRATCH_GB = float(os.environ.get("OPAQUE_SCRATCH_GB", "200"))
DEFAULT_TMP_GB = float(os.environ.get("OPAQUE_TMP_GB", "100"))
SCRATCH_DIR = os.environ.get("OPAQUE_SCRATCH_DIR", "/scratch")

# Orchestrator pod resources. The orchestrator only spawns the step Job and
# talks to the ZenML server, so it gets a lean CPU-only slice (no GPU, no
# scratch PVCs, no creds). Crucially it must NOT inherit the step's H100 request
# -- otherwise it sits unschedulable waiting for a *second* GPU just to
# coordinate, doubling the scarce-H100 demand of every run.
DEFAULT_ORCH_MEMORY_GB = float(os.environ.get("OPAQUE_ORCH_MEMORY_GB", "4"))
DEFAULT_ORCH_MEMORY_LIMIT_GB = float(os.environ.get("OPAQUE_ORCH_MEMORY_LIMIT_GB", "8"))
DEFAULT_ORCH_CPU_COUNT = float(os.environ.get("OPAQUE_ORCH_CPU_COUNT", "2"))

# Both pods run the image's non-numeric ``nonroot`` user, which the non-TRACE
# stacks' ``runAsNonRoot`` admission check rejects unless a numeric UID is
# pinned. Pin UID/GID 1000 (the ``nonroot`` user) so the check passes on the
# orchestrator AND the step pod; ``fsGroup`` additionally makes the step's
# root-owned /scratch + /tmp PVCs group-writable by that user.
_SECURITY_CONTEXT: dict[str, Any] = {
    "security_context": {
        "runAsUser": 1000,
        "runAsGroup": 1000,
        "runAsNonRoot": True,
        "fsGroup": 1000,
    }
}


def _gpu_type() -> GPUs:
    """Resolve ``OPAQUE_GPU_TYPE`` to a ``GPUs`` enum member (default H100)."""
    try:
        return GPUs[DEFAULT_GPU_TYPE]
    except KeyError as exc:
        valid = ", ".join(g.name for g in GPUs)
        raise ValueError(f"OPAQUE_GPU_TYPE={DEFAULT_GPU_TYPE!r} is not one of: {valid}") from exc


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
    """Kubernetes + Docker settings for an opaque training run.

    Returns a single pipeline-level ``settings`` block that drives two pods with
    different specs: the GPU trainer (``pod_settings``) and a lean CPU-only
    orchestrator (``orchestrator_pod_settings``) that only spawns the step Job.
    Keeping them separate stops the orchestrator from grabbing a second H100.

    Args:
        gpu: request an H100 for the step (``False`` runs CPU-only, e.g. the
            tiny-gpt2 smoke); the orchestrator is always CPU-only.
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

    # Step (trainer) pod: the actual GPU work. Mounts /scratch + /tmp PVCs, gets
    # the W&B/HF creds, and requests the H100. ``dict(_SECURITY_CONTEXT)`` is
    # copied because ``get_pod_config`` mutates ``additional_pod_spec_args`` in
    # place (it may append init containers), so the step and orchestrator must
    # not share one dict.
    step_pod_configuration = PodConfiguration(
        memory_gb=DEFAULT_MEMORY_GB,
        memory_limit_gb=DEFAULT_MEMORY_LIMIT_GB,
        cpu_count=DEFAULT_CPU_COUNT,
        ephemeral_storage_gb=5,
        gpu=_gpu_type() if gpu else None,
        gpu_count=1 if gpu else None,
        extra_envs=extra_envs,
        storage_configuration=[
            MountConfiguration(mount_path=SCRATCH_DIR, size_gb=DEFAULT_SCRATCH_GB),
            MountConfiguration(mount_path="/tmp", size_gb=DEFAULT_TMP_GB),
        ],
        additional_pod_spec_args=dict(_SECURITY_CONTEXT),
    )

    # Orchestrator pod: lean, CPU-only (no GPU/PVCs/creds). Only the numeric-UID
    # security context is shared with the step. This is what keeps the
    # orchestrator off the H100 -- see DEFAULT_ORCH_* above.
    orchestrator_pod_configuration = PodConfiguration(
        memory_gb=DEFAULT_ORCH_MEMORY_GB,
        memory_limit_gb=DEFAULT_ORCH_MEMORY_LIMIT_GB,
        cpu_count=DEFAULT_ORCH_CPU_COUNT,
        ephemeral_storage_gb=2,
        extra_envs=[{"name": "PYTHONUNBUFFERED", "value": "1"}],
        additional_pod_spec_args=dict(_SECURITY_CONTEXT),
    )

    # Submit asynchronously, exactly like NES (`pusk/launch.py` forces
    # ``synchronous=False`` on every run): the client returns as soon as the
    # orchestrator Job is created instead of blocking for the whole run, and pod
    # startup/retries are left to the Kubernetes Job. The run is then monitored
    # via the dashboard / ZenML API.
    orchestrator_kwargs = {"synchronous": False, **kwargs.pop("orchestrator_kwargs", {})}
    # Optional: keep finished/failed Jobs (and their pods/events) around for
    # post-mortem inspection instead of the stack's aggressive default TTL.
    if (ttl := os.environ.get("OPAQUE_JOB_TTL_SECONDS")):
        orchestrator_kwargs.setdefault("ttl_seconds_after_finished", int(ttl))

    image = get_image_url(DockerImage.TRAIN, tag=docker_image_tag)

    # Two calls so the step and orchestrator pods get *different* pod specs from
    # one pipeline-level ``settings`` block: ``is_for_orchestrator=False`` makes
    # the first call set only ``pod_settings`` (the GPU step), and
    # ``is_for_step=False`` makes the second set only ``orchestrator_pod_settings``
    # (the lean orchestrator). jb-mlops otherwise applies one pod spec to both.
    step_settings = get_step_settings(
        docker_image=image,
        pod_configuration=step_pod_configuration,
        orchestrator_kwargs=orchestrator_kwargs,
        is_for_step=True,
        is_for_orchestrator=False,
        **kwargs,
    )
    orchestrator_settings = get_step_settings(
        docker_image=image,
        pod_configuration=orchestrator_pod_configuration,
        orchestrator_kwargs=orchestrator_kwargs,
        is_for_step=False,
        is_for_orchestrator=True,
    )
    # Fold the lean orchestrator pod settings into the step settings so a single
    # pipeline-level ``settings`` drives both pods.
    step_settings["orchestrator"] = step_settings["orchestrator"].model_copy(
        update={
            "orchestrator_pod_settings": orchestrator_settings["orchestrator"].orchestrator_pod_settings
        }
    )
    return step_settings
