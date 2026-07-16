"""Runtime Docker image resolution for opaque ZenML steps.

Mirrors ``next-edit-pipeline``'s ``jetbrains/nes/zenml/docker_images.py``: a
small enum of prebuilt images plus a resolver that turns
``<registry>/<project>-<image>:<tag>`` into a full URL, with a per-image env
override so a custom branch can point one step at a freshly-built tag without
editing code.

The image itself is prebuilt and pushed to a GCP Artifact Registry the target
ZenML GPU cluster can pull from (see ``build_and_push.sh``); ZenML consumes it
with ``skip_build=True`` (the pod never builds anything).
"""

from __future__ import annotations

import os
from enum import Enum

__all__ = ["DockerImage", "get_image_url", "get_override_env_name"]

# Registry the target GKE cluster pulls from. Defaults to the GCP Artifact
# Registry opaque's CI Workload Identity can push to (same project as the
# devcontainer image). Override with OPAQUE_DOCKER_REGISTRY — e.g. point at
# ``europe-west4-docker.pkg.dev/grazie-development/grazie-ml`` (where NES pushes)
# if the cluster can't pull from this project.
DOCKER_REGISTRY = os.environ.get(
    "OPAQUE_DOCKER_REGISTRY",
    "europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml",
)
PROJECT_NAME = "opaque"
DOCKER_TAG = os.environ.get("OPAQUE_DOCKER_TAG", "latest")


class DockerImage(Enum):
    """Prebuilt images available to opaque ZenML steps."""

    TRAIN = "train"


def get_override_env_name(image: DockerImage) -> str:
    """Per-image full-URL override env var, e.g. ``OPAQUE_DOCKER_IMAGE_TRAIN``."""
    suffix = image.value.replace("-", "_").upper()
    return f"OPAQUE_DOCKER_IMAGE_{suffix}"


def get_image_url(image: DockerImage, tag: str | None = None) -> str:
    """Full image URL, e.g. ``<registry>/opaque-train:latest``.

    A full-URL override in ``OPAQUE_DOCKER_IMAGE_<IMAGE>`` wins over the
    registry/tag composition.
    """
    if override := os.environ.get(get_override_env_name(image), "").strip():
        return override
    tag = tag or DOCKER_TAG
    return f"{DOCKER_REGISTRY}/{PROJECT_NAME}-{image.value}:{tag}"
