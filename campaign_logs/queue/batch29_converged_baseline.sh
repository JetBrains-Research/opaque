#!/usr/bin/env bash
# BATCH 29 — rerun the two 1560-step momentum-correct baselines that crashed.
#
# q-norot-mom-e6-s42 crashed at step 1270/1560 (loss 0.696377) and s43 at 1360
# (0.696177). Partial reads only. The converged comparison is the one that shows
# whether the 4.77e-3 at 520 steps persists: at 520 the confounded version showed
# 8.86e-3 and the converged confounded version 7.90e-3, so ~89% retention. If the
# clean gap retains similarly the converged number is near 4.2e-3.
#
# Mid-flight the crashed runs were at 0.6963 against rotation's 0.692347, i.e. a
# gap near 3.9e-3 at ~85% of training, so the retention looks similar. Confirm it.
#
# Third seed added because the pair is the paper's converged headline.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-dfb3f72
MAXCONC="${MAXCONC:-5}"; LOGDIR=campaign_logs/batch29; mkdir -p "$LOGDIR"
Q="--lora-r 16 --lora-alpha 16 --weight-decay 0 --eval-bpb --eval-batch-size 16 \
--optimizer sgd --sgd-momentum 0.9 --learning-rate 5e-2 --lora-xse-p-e 0 --num-epochs 6"
QUEUE=("q-norot-mom-e6b-s42|42" "q-norot-mom-e6b-s43|43" "q-norot-mom-e6b-s44|44")
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
for spec in "${QUEUE[@]}"; do
  IFS='|' read -r NAME SEED <<< "$spec"
  [[ "$(exists "$NAME")" == "yes" ]] && { echo "[skip] $NAME"; continue; }
  while :; do
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC cap=$MAXCONC zenml=$ZC — holding $NAME"; sleep 180
  done
  echo "[submit] $NAME seed=$SEED"
  OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
    --run-name "$NAME" --seed "$SEED" --extra $Q > "$LOGDIR/$NAME.log" 2>&1
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 200
done
echo "BATCH 29 QUEUED"
