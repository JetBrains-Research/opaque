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

WORKDIR /opt/opaque
COPY . /opt/opaque

# Resolve the workspace (builds opaque-accounting), then add the trainer's
# runtime deps and a ZenML client matching the server (0.94.6). We install the
# training deps explicitly rather than via `--group examples` to avoid the
# internal-only cadence/keyring/jupyter deps that aren't needed in-cluster.
# (--eval-humaneval/--eval-mbpp additionally need `evalplus`; the renyi arms
# don't use them, so it's omitted to keep the image small.)
RUN uv sync --all-packages --extra all \
    && uv pip install \
        "torchopt>=0.7.3" \
        "datasets>=2.0.0" \
        "transformers==4.57.1" \
        "peft>=0.18.0" \
        "wandb>=0.16.0" \
        ./vendor/lora-privacy \
        "zenml==0.94.6" \
        "kubernetes" \
        "gcsfs" \
        "google-cloud-storage"

# Make the workspace venv the default interpreter so ZenML (and the trainer
# subprocess) run inside it.
ENV VIRTUAL_ENV=/opt/opaque/.venv \
    PATH="/opt/opaque/.venv/bin:${PATH}" \
    OPAQUE_REPO_DIR=/opt/opaque

# Sanity check: opaque (with its Rust extension) and torch import cleanly.
RUN python -c "import opaque.accounting, torch, transformers, peft, zenml; print('opaque image OK', torch.__version__)"
