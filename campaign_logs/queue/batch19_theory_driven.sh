#!/usr/bin/env bash
# BATCH 19 — the two theory-driven interventions that were IMPLEMENTED AND NEVER RUN.
# Both shipped in image 11b5a0c weeks ago. Each is predicted by a specific piece of
# theory, so each is a test rather than a sweep.
#
# (A) GRADIENT-BASIS INIT (--lora-xs-init grad). LoRA-XS freezes B, A from the SVD
#     of the pretrained weight W0. But to first order the loss decrease from an
#     update dW is <-grad, dW>, so the rank-r subspace that maximises it is set by
#     the GRADIENT, not by W0 -- W0's top subspace is only a proxy, good exactly to
#     the extent the gradient aligns with it. The paper's own ablation shows the
#     proxy failing on SST-2 (SVD-of-random beat SVD-of-W0), which they attribute
#     to the task being far from the pretraining objective. If the theory is right,
#     a gradient basis should beat a W0 basis at IDENTICAL parameter count -- an
#     attack on the method's core design decision, not a hyperparameter.
#     Two arms so the init is separated from the rotation.
#
#     History: this failed twice before reaching the cluster. First a TypeError
#     (collate returns a 1-tuple, not a dict), then CUDA OOM from holding 196
#     simultaneous gradient buffers. Now rewritten with post-accumulate-grad hooks
#     reducing each gradient to rank-r factors in place (114k numbers instead of
#     12.8M) and probe batch_size 1. Verified on a 6-layer model: all layers
#     rebased, ||B^T B - I|| = 9.3e-7, base weights restored, no leftover .grad.
#
# (B) SCALAR PRECONDITIONING (XSE_ADAM_PRECOND=scalar). Our no-go theorem says a
#     diagonal preconditioner commutes with the two-sided basis change iff it is
#     constant on the connected components of supp(K), K = Rt^T (kron) L; a dense K
#     forces D = cI. So SCALAR is the UNIQUE equivariant choice -- the impossibility
#     proof hands over its own fix.
#
#     Honest prior: this will probably TIE, not win. transport-vs-carry measured a
#     null (5.16e-5, below the 6.23e-5 seed sd), and the contraction result says
#     rotation drives the preconditioner condition number 11.53 -> 3.33 on its own,
#     so the diagonal barely discriminates directions anyway. A tie is still a
#     result: it confirms the contraction prediction and makes the impossibility
#     theorem consequential rather than decorative. A LOSS would falsify the
#     contraction story, which is why it is worth the GPU time.
#
# BASELINES these compare against (same image family, 520 steps, r=16):
#   frozen  + SGD   0.702079 (n=3)      rotation + SGD   0.693218 (n=3)
#   frozen  + AdamW 0.695331 (n=3)      rotation + AdamW 0.692545 (diag, lr 2e-3)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-11b5a0c
MAXCONC="${MAXCONC:-4}"; LOGDIR=campaign_logs/batch19; mkdir -p "$LOGDIR"
C="--lora-r 16 --num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16"
ROT="--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"
GRAD="--lora-xs-init grad --lora-xs-init-batches 1"

# name | env (or -) | args
QUEUE=(
  "sb-grad-norot-s42|-|--optimizer sgd --learning-rate 5e-2 --lora-xse-p-e 0 $GRAD $C"
  "sb-grad-xse-d5t1-s42|-|--optimizer sgd --learning-rate 5e-2 $ROT $GRAD $C"
  "scalar-xse-d5t1-lr2e3-s42|XSE_ADAM_PRECOND=scalar|--optimizer adamw --learning-rate 2e-3 $ROT $C"
  "scalar-norot-lr1e3-s42|XSE_ADAM_PRECOND=scalar|--optimizer adamw --learning-rate 1e-3 --lora-xse-p-e 0 $C"
)
running_count() { uv run python - <<'PY' 2>/dev/null
import os
os.environ.setdefault("WANDB_BASE_URL","https://jetbrains.wandb.io")
import wandb
print(len(list(wandb.Api(timeout=60).runs("federated-compute/opaque-lora-xs",filters={"state":"running"}))))
PY
}
exists() { RUN_NAME="$1" uv run python - <<'PY' 2>/dev/null
import os
os.environ.setdefault("WANDB_BASE_URL","https://jetbrains.wandb.io")
import wandb
print("yes" if list(wandb.Api(timeout=60).runs("federated-compute/opaque-lora-xs",
      filters={"display_name":os.environ["RUN_NAME"]})) else "no")
PY
}
img_ready() { gcloud artifacts docker tags list "${OPAQUE_DOCKER_REGISTRY}/opaque-train" \
    --format='value(tag)' 2>/dev/null | grep -qx "$1"; }
for spec in "${QUEUE[@]}"; do
  IFS='|' read -r NAME ENVS ARGS <<< "$spec"
  [[ "$(exists "$NAME")" == "yes" ]] && { echo "[skip] $NAME"; continue; }
  while :; do
    if ! img_ready "$IMG"; then echo "[wait] image missing"; sleep 300; continue; fi
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC cap=$MAXCONC zenml=$ZC — holding $NAME"; sleep 300
  done
  echo "[submit] $NAME env=[$ENVS]"
  if [[ "$ENVS" == "-" ]]; then
    OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
      --run-name "$NAME" --seed 42 --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  else
    env $ENVS OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
      --run-name "$NAME" --seed 42 --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  fi
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 300
done
echo "BATCH 19 QUEUED"
