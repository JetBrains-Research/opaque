# Training image for running the Rényi DP experiments on a ZenML Kubernetes GPU
# stack. Installs the opaque uv workspace (incl. the Rust `opaque-accounting`
# build) plus the trainer's runtime deps and a matching ZenML client.
#
# Build from the REPO ROOT so the uv workspace + vendored submodule are in
# context:  docker build -f deploy/zenml/Dockerfile -t <image> .
#
# torch is pinned <2.11 (see pyproject [tool.uv] constraint) to keep CUDA-12
# runtime deps, matching this CUDA 12.4 base.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# uv (Python/deps) + Rust toolchain (maturin build of opaque-accounting)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# The ZenML Kubernetes stack rejects root containers (the kubelet fails them with
# `CreateContainerConfigError: container has runAsNonRoot and image will run as
# root`). Create an unprivileged runtime user up-front, matching the jb-mlops /
# NES convention (user `nonroot`, home `/home/nonroot`). `/app` is where the
# ZenML entrypoint writes its config + downloaded pipeline code
# (`ZENML_CONFIG_PATH=/app/.config`).
ARG NONROOT_UID=1000
ARG NONROOT_GID=1000
RUN groupadd --gid ${NONROOT_GID} nonroot \
    && useradd --create-home --home-dir /home/nonroot --uid ${NONROOT_UID} --gid ${NONROOT_GID} --shell /bin/bash nonroot \
    && mkdir -p /app

WORKDIR /opt/opaque
COPY . /opt/opaque

# Resolve the workspace (builds opaque-accounting), then add the trainer's
# runtime deps and a ZenML client matching the server (0.94.6). We install the
# training deps explicitly rather than via `--group examples` to avoid the
# internal-only cadence/keyring/jupyter deps that aren't needed in-cluster.
# `evalplus` (below) is included so --eval-humaneval/--eval-mbpp downstream
# pass@1 evaluation works in-cluster (the utility-comparison arms need it).
#
# `zenml[connectors-gcp]` (not the full `zenml integration install gcp`) pulls the
# minimal prerequisites for ZenML's GCP service connector — google-cloud-container
# + google-cloud-artifact-registry (+ storage/kubernetes we already need). The
# GCS artifact store authenticates via that connector, so without these the step
# pod dies with `NotImplementedError: Service connector type gcp is not available
# locally`. We deliberately avoid the full integration (kfp / aiplatform /
# pipeline-components + a gcsfs<=2024.12 downgrade) to keep the image lean and
# steer clear of dependency conflicts with torch/transformers. `gcsfs` is the one
# extra the artifact store's filesystem needs that the connector extra omits.
#
# Ownership of the app dirs is handed to the runtime user in this same layer so
# the multi-GB venv isn't duplicated into a separate `chown` layer.
RUN uv sync --all-packages --extra all \
    && uv pip install \
        "torchopt>=0.7.3" \
        "datasets>=2.0.0" \
        "transformers==4.57.1" \
        "peft>=0.18.0" \
        "wandb>=0.16.0" \
        ./vendor/lora-privacy \
        "zenml[connectors-gcp]==0.94.6" \
        "gcsfs" \
        "evalplus==0.3.1" \
    && chown -R ${NONROOT_UID}:${NONROOT_GID} /opt/opaque /app /home/nonroot

# Make the workspace venv the default interpreter so ZenML (and the trainer
# subprocess) run inside it, and point HOME at the nonroot home. The trainer's
# cwd is /opt/opaque (owned by nonroot above); large caches/temp are routed onto
# the mounted /scratch volume at runtime (see pipeline.py).
ENV VIRTUAL_ENV=/opt/opaque/.venv \
    PATH="/opt/opaque/.venv/bin:${PATH}" \
    OPAQUE_REPO_DIR=/opt/opaque \
    HOME=/home/nonroot

USER nonroot

# Sanity check: opaque (with its Rust extension) and torch import cleanly as the
# unprivileged runtime user, and the ZenML GCP service connector registers (it is
# what the GCS artifact store uses to authenticate from the step pod). Failing
# here surfaces a missing connector dep at build time instead of mid-run.
RUN python -c "import opaque.accounting, torch, transformers, peft, zenml; \
from zenml.service_connectors.service_connector_registry import service_connector_registry as r; \
r.register_builtin_service_connectors(); \
assert 'gcp' in r.service_connector_types, 'GCP service connector failed to register'; \
print('opaque image OK', torch.__version__, '| gcp connector OK')"
