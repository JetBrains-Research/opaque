# Running the Rényi DP experiments on ZenML (non-TRACE GPU)

This directory runs the two-arm Rényi effective-rank experiment from
[`docs/renyi-dp-experiment-handoff.md`](../../docs/renyi-dp-experiment-handoff.md)
on a **non-TRACE** Kubernetes GPU stack, instead of Cadence.

It deliberately **mirrors how `next-edit-pipeline` (NES) deals with ZenML**:

- step/pod settings are built with the shared **`jb-mlops`**
  `jetbrains.mlops.zenml.get_step_settings` helper (same one NES training uses),
- it targets the same non-TRACE resources NES training defaults to — project
  **`models-rd`**, stack **`gke-ai-for-code`**, GPU **H100 ×1**,
- W&B / HF credentials come from the **existing shared `ai-for-code` ZenML
  secret** (reused verbatim — no opaque-specific secret needed),
- a prebuilt CUDA image is pushed to a GCP Artifact Registry and consumed with
  `skip_build=True` (the pod never builds anything).

Each pipeline run shells out to `examples/train_causal_lm.py` with the exact
flags from the handoff doc / `.cadence/configs/renyi_dp_vs_nodp.yaml`, so results
are identical to a local/Cadence launch — you're just borrowing ZenML GPU time.

| File | Purpose |
| --- | --- |
| `docker_images.py` | Image enum + `<registry>/opaque-train:<tag>` resolver (mirrors NES `docker_images.py`) |
| `settings.py` | `training_settings()` → jb-mlops `get_step_settings` (H100 pod, secret env, image) |
| `pipeline.py` | ZenML `@pipeline` + one `@step` that runs the trainer CLI (pod-side; `zenml`+stdlib only) |
| `run.py` | Submits arms (`smoke`/`dp`/`nodp`/`both`); activates project/stack; attaches settings |
| `Dockerfile` | CUDA-12 image: opaque workspace (incl. Rust `opaque-accounting`) + trainer deps + ZenML |
| `build_and_push.sh` | Build & push that image to the Artifact Registry |
| `requirements.txt` | Client-side submit deps (`zenml`, `jb-mlops[zenml]`, k8s/gcs) |

## What runs

| Arm | Trainer flags | GPU |
| --- | --- | --- |
| `smoke` | `--preset custom --model-name sshleifer/tiny-gpt2 --num-train-samples 512 --num-epochs 1 --lora-method lora-xs --lora-xse-p-e 0.333 --optimizer sgd --sgd-momentum 0.9` | no (`--no-gpu`) |
| `dp` | `--preset qwen-coder-kstack-lora --lora-method lora-xs --lora-xse-p-e 0.333 --num-epochs 1` | H100 |
| `nodp` | …same as `dp` plus `--noise-multiplier 0` | H100 |

Both real arms log to W&B `federated-compute/opaque-lora-xs` (metrics
`rotation/r_eff_*`, `rotation/renyi_gap_a0p5_ainf`).

## Prerequisites (one-time)

Everything here is **non-TRACE** — no AWS role, no `trace` project.

1. **ZenML login.** `zenml login <server>` once (or set `OPAQUE_ZENML_LOGIN`).
   The defaults submit to project `models-rd` + stack `gke-ai-for-code`.
2. **`jb-mlops` client.** It lives on the private `space-tools` index; install
   with `uv` (which reads `UV_INDEX_SPACE_TOOLS_USERNAME/PASSWORD` from your env):

   ```bash
   uv venv .zenml-client && source .zenml-client/bin/activate
   uv pip install \
     --index space-tools=https://packages.jetbrains.team/pypi/p/grazi/jetbrains-ai-tools/simple \
     -r deploy/zenml/requirements.txt
   ```

3. **A registry the `gke-ai-for-code` cluster can pull from.** CI defaults to
   `europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml` (the project opaque's
   Workload Identity can push to — same as the devcontainer image). If the
   cluster can't pull from there, point `OPAQUE_DOCKER_REGISTRY` at
   `…/grazie-development/grazie-ml` (where NES pushes) instead.
4. The shared **`ai-for-code`** ZenML secret already holds `WANDB_API_KEY` +
   `HF_TOKEN`; nothing to create. (Override with `OPAQUE_ZENML_SECRET` if you
   keep creds elsewhere.)

### Build & push the training image

**Recommended: GitHub Actions** (`.github/workflows/build-train-image.yml`).
CI runners are Linux `amd64`, so the heavy CUDA/Rust build avoids a laptop
QEMU cross-build and gets layer caching. It builds `deploy/zenml/Dockerfile`
and pushes `opaque-train:<branch>-<sha>` on push to `main` or the
`david-stan/zenml-training` branch (auth via opaque's existing Workload
Identity — no SA key needed).

- **Only secret required:** **`SUBMODULE_PAT`** — a PAT with read access to
  `JetBrains-Research/LoRA-Privacy`, because `vendor/lora-privacy` is a separate
  private repo the default Actions token can't fetch. GCP auth reuses the repo
  vars `GCP_WORKLOAD_IDENTITY_PROVIDER` + `GCP_SERVICE_ACCOUNT_EMAIL` (already
  set for the devcontainer build).
- Optional repo vars: `OPAQUE_DOCKER_REGISTRY`, `TRAIN_IMAGE_RUNNER` (a bigger
  runner if `ubuntu-latest` runs out of disk).
- The run summary prints the exact `OPAQUE_DOCKER_TAG` to submit with.

**Or build locally:**

```bash
OPAQUE_DOCKER_REGISTRY=europe-west4-docker.pkg.dev/<project>/<repo> \
  OPAQUE_DOCKER_TAG=$(git rev-parse --short HEAD) \
  ./deploy/zenml/build_and_push.sh

# then point run.py at the same tag:
export OPAQUE_DOCKER_REGISTRY=europe-west4-docker.pkg.dev/<project>/<repo>
export OPAQUE_DOCKER_TAG=$(git rev-parse --short HEAD)
```

## Run it

```bash
# 0) resolve everything without submitting (works even before the image exists)
python deploy/zenml/run.py dp --dry-run

# 1) cheap wiring check (tiny-gpt2, CPU) — confirms image/secret/upload and that
#    rotation/renyi_gap_a0p5_ainf shows up in W&B
python deploy/zenml/run.py smoke --no-gpu

# 2) the real pair (Qwen2.5-Coder-7B, H100)
python deploy/zenml/run.py both
#    ...or individually:
python deploy/zenml/run.py dp
python deploy/zenml/run.py nodp
```

For the paper figure, run a few seeds:

```bash
for s in 42 43 44; do
  python deploy/zenml/run.py dp   --seed "$s" --run-name "renyi-dp-eps3-s$s"
  python deploy/zenml/run.py nodp --seed "$s" --run-name "renyi-nodp-s$s"
done
```

## Configuration knobs (env vars)

| Env var | Default | Meaning |
| --- | --- | --- |
| `OPAQUE_ZENML_PROJECT` | `models-rd` | ZenML project to submit to |
| `OPAQUE_ZENML_STACK` | `gke-ai-for-code` | non-TRACE GPU stack |
| `OPAQUE_ZENML_LOGIN` | *(none)* | if set, `zenml login <target>` before submit |
| `OPAQUE_DOCKER_REGISTRY` | `…/gke-dev-dws-jbr/ml` | registry for `opaque-train` (WIF can push; override to grazie-ml if needed) |
| `OPAQUE_DOCKER_TAG` | `latest` | image tag |
| `OPAQUE_DOCKER_IMAGE_TRAIN` | *(none)* | full image URL override (wins over registry/tag) |
| `OPAQUE_ZENML_SECRET` | `ai-for-code` | ZenML secret with `WANDB_API_KEY` + `HF_TOKEN` |
| `OPAQUE_MEMORY_GB` / `OPAQUE_MEMORY_LIMIT_GB` | `160` / `200` | pod memory request / limit |
| `OPAQUE_CPU_COUNT` | `20` | pod CPU request |
| `OPAQUE_SCRATCH_GB` / `OPAQUE_TMP_GB` | `200` / `100` | scratch (HF cache/tmp) volume sizes |
| `OPAQUE_SCRATCH_DIR` | `/scratch` | mount point for caches/tmp |
| `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_BASE_URL` | `opaque-lora-xs` / `federated-compute` / `https://jetbrains.wandb.io` | W&B target |

## How this maps to NES

| NES (`jetbrains/nes/zenml/…`) | Here (`deploy/zenml/…`) |
| --- | --- |
| `docker_images.py` (registry/tag + `NES_DOCKER_IMAGE_*` override) | `docker_images.py` (`OPAQUE_DOCKER_*`) |
| `base.py` + `pipelines/training/settings.py` (`get_step_settings`, `PodConfiguration(gpu=GPUs.H100)`, `wandb_envs`/`hf_envs` off `ai-for-code`) | `settings.py` (same helpers, same secret) |
| `pipelines/training/sft_pipeline.py` (`@zenml.pipeline` + training `@step`) | `pipeline.py` |
| `pusk` launcher (`execute_run_request`: set project, activate stack, submit) | `run.py` (trimmed to submit + arms) |
| `docker/Dockerfile` (parametrized CUDA image → GCP AR) | `Dockerfile` + `build_and_push.sh` |

The main simplification: opaque's trainer is a CLI, so the step invokes it as a
subprocess (as the handoff doc suggests) rather than reimplementing NES's
in-process `train()`.

## Troubleshooting

- **Pod stuck `Pending`** — no H100 capacity in `gke-ai-for-code`; jb-mlops sets
  the GPU node selector/tolerations, so this is usually capacity, not config.
- **`ImagePullBackOff`** — the cluster can't pull the image; push it to a
  registry it can access and set `OPAQUE_DOCKER_REGISTRY`/`OPAQUE_DOCKER_TAG`.
- **`ModuleNotFoundError: jetbrains.mlops`** — install `jb-mlops[zenml]` from
  `space-tools` (see prereqs); it's a submit-side dep only.
- **W&B offline / 401** — the `ai-for-code` secret must expose `WANDB_API_KEY`
  and `HF_TOKEN` (it does for NES); check `zenml secret get ai-for-code`.
- **Trainer edits not reflected** — the pod runs the repo baked into the image
  at `/opt/opaque`; rebuild/push (or set `OPAQUE_REPO_DIR`) to pick up changes.
```
