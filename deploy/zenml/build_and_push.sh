#!/usr/bin/env bash
# Build the opaque training image and push it to the GCP Artifact Registry the
# target ZenML GPU stack (gke-ai-for-code) pulls from.
#
# The image URL is composed exactly like deploy/zenml/docker_images.py:
#   ${OPAQUE_DOCKER_REGISTRY}/opaque-train:${OPAQUE_DOCKER_TAG}
# or override the whole thing with OPAQUE_DOCKER_IMAGE_TRAIN.
#
#   OPAQUE_DOCKER_REGISTRY=europe-west4-docker.pkg.dev/<proj>/<repo> \
#     OPAQUE_DOCKER_TAG=$(git rev-parse --short HEAD) \
#     ./deploy/zenml/build_and_push.sh
set -euo pipefail

# BuildKit so the per-Dockerfile ignore (deploy/zenml/Dockerfile.dockerignore)
# is honored with the repo-root build context.
export DOCKER_BUILDKIT=1

REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/grazie-development/grazie-ml}"
TAG="${OPAQUE_DOCKER_TAG:-latest}"
IMAGE="${OPAQUE_DOCKER_IMAGE_TRAIN:-${REGISTRY}/opaque-train:${TAG}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# The Rust accounting build + LoRA-XSe live in the submodule; make sure it's here.
git submodule update --init --recursive

# Configure docker auth for the registry host (Artifact Registry).
REGISTRY_HOST="${IMAGE%%/*}"
if command -v gcloud >/dev/null 2>&1 && [[ "$REGISTRY_HOST" == *docker.pkg.dev ]]; then
    gcloud auth configure-docker "$REGISTRY_HOST" --quiet
fi

docker build --platform linux/amd64 -f deploy/zenml/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"
echo "pushed $IMAGE"
echo
echo "Point the pipeline at it (matches deploy/zenml/docker_images.py):"
echo "  export OPAQUE_DOCKER_REGISTRY=$REGISTRY OPAQUE_DOCKER_TAG=$TAG"
echo "  # or: export OPAQUE_DOCKER_IMAGE_TRAIN=$IMAGE"
