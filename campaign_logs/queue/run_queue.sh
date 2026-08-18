#!/usr/bin/env bash
# Staged submitter: submits the queued runs, never exceeding MAXCONC concurrent
# GPU runs. Over-subscribing the ~5-slot quota co-locates pods and crashed 7 runs
# in July, so the cap is not optional.
#
#   host zenml.labs.jb.gg && curl -s -o /dev/null -w '%{http_code}\n' \
#     --max-time 10 https://zenml.labs.jb.gg/api/v1/info      # expect 200 FIRST
#   WANDB_BASE_URL=https://jetbrains.wandb.io ./campaign_logs/queue/run_queue.sh
#
# Idempotent: a run whose name already exists in W&B is skipped, so re-running
# this after a VPN drop resumes rather than duplicating.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export OPAQUE_DOCKER_TAG="${OPAQUE_DOCKER_TAG:-david-stan-zenml-training-7e97389}"
export WANDB_MODE=online
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"

MAXCONC="${MAXCONC:-4}"
LOGDIR=campaign_logs/queue
mkdir -p "$LOGDIR"

# name | XSE env (or "-") | trainer --extra args
QUEUE=(
  # --- (1) FLOOR: replicates of a non-adaptive DEEP config. The denominator of
  #         every effect size in the paper; currently n=1 at this depth.
  "floor-fixed-re13-s43|-|--lora-xse-p-e 0.8125 --lora-r 16 --num-epochs 1 --seed 43"
  "floor-fixed-re13-s44|-|--lora-xse-p-e 0.8125 --lora-r 16 --num-epochs 1 --seed 44"
  "floor-fixed-re13-s45|-|--lora-xse-p-e 0.8125 --lora-r 16 --num-epochs 1 --seed 45"
  "floor-fixed-re13-s46|-|--lora-xse-p-e 0.8125 --lora-r 16 --num-epochs 1 --seed 46"

  # --- (2) 2-EPOCH DEPTH CURVE: attacks the ROOT CAUSE of the noise floor.
  #         Eval loss is still falling at step 260, which is why last-k averaging
  #         fails and loss_min is discontinuous. Train to convergence and the
  #         floor should drop. Also matches the headline config (2 epochs).
  "ep2-fixed-re1-s42|-|--lora-xse-p-e 0.0625 --lora-r 16 --num-epochs 2"
  "ep2-fixed-re5-s42|-|--lora-xse-p-e 0.3125 --lora-r 16 --num-epochs 2"
  "ep2-fixed-re9-s42|-|--lora-xse-p-e 0.5625 --lora-r 16 --num-epochs 2"
  "ep2-fixed-re13-s42|-|--lora-xse-p-e 0.8125 --lora-r 16 --num-epochs 2"

  # --- (3) RANK-64 SPECTRUM PROBE, 1 run, measurement only, no utility claim.
  #         Prediction: p_1 stays ~0.8 => floor(N_inf)=1 => the collapse holds at
  #         r=64 too, widening Theorem 2 from "r=16 here" to "any rank where the
  #         gradient is rank-1 dominant".  lora-alpha=64 keeps alpha/r=1 as at r=16.
  "r64-probe-a1-m2-s42|XSE_ADAPTIVE_DEPTH=1 XSE_ADAPTIVE_DEPTH_ALPHA=1 XSE_ADAPTIVE_DEPTH_MARGIN=2|--lora-xse-p-e 0.333 --lora-r 64 --lora-alpha 64 --num-epochs 1"

  # --- (4) EVAL-PHASE PROBE. Rotation fires every 5 steps and eval every 10, so
  #         EVERY eval currently lands on a rotation step. eval-steps 7 is coprime
  #         to 5, so evals sample all phases -- the only way to see a per-rotation
  #         cost on EVAL loss. Training dynamics are untouched; only when we look.
  "phase-eval7-nodp-s42|-|--lora-xse-p-e 0.333 --lora-r 16 --num-epochs 1 --eval-steps 7"
)

running_count() {
  uv run python - <<'PY' 2>/dev/null
import os
os.environ.setdefault("WANDB_BASE_URL", "https://jetbrains.wandb.io")
import wandb
print(len(list(wandb.Api(timeout=60).runs("federated-compute/opaque-lora-xs",
                                          filters={"state": "running"}))))
PY
}

exists() {
  RUN_NAME="$1" uv run python - <<'PY' 2>/dev/null
import os
os.environ.setdefault("WANDB_BASE_URL", "https://jetbrains.wandb.io")
import wandb
nm = os.environ["RUN_NAME"]
n = len(list(wandb.Api(timeout=60).runs("federated-compute/opaque-lora-xs",
                                        filters={"display_name": nm})))
print("yes" if n else "no")
PY
}

for spec in "${QUEUE[@]}"; do
  NAME="${spec%%|*}"; rest="${spec#*|}"; ENVS="${rest%%|*}"; ARGS="${rest#*|}"

  if [[ "$(exists "$NAME")" == "yes" ]]; then
    echo "[skip] $NAME already in W&B"; continue
  fi

  while :; do
    RC="$(running_count)"
    [[ -z "$RC" ]] && { echo "[warn] cannot reach W&B; waiting"; sleep 120; continue; }
    (( RC < MAXCONC )) && break
    echo "[wait] $RC running (cap $MAXCONC) — holding $NAME"; sleep 300
  done

  echo "[submit] $NAME  env=[$ENVS]  args=[$ARGS]"
  if [[ "$ENVS" == "-" ]]; then
    .zenml-client/bin/python deploy/zenml/run.py nodp --run-name "$NAME" \
      --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  else
    env $ENVS .zenml-client/bin/python deploy/zenml/run.py nodp --run-name "$NAME" \
      --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  fi
  if grep -q "^submitted" "$LOGDIR/$NAME.log"; then
    echo "[ok] $NAME"
  else
    echo "[FAIL] $NAME — see $LOGDIR/$NAME.log (tail below)"
    tail -3 "$LOGDIR/$NAME.log"
  fi
  sleep 20
done
echo "QUEUE DONE"
