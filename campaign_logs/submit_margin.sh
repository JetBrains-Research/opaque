#!/usr/bin/env bash
# Margin sweep: m in {0,4,6} at alpha=1, non-DP, seed 42.
# (m=1,2,3 already exist -> completes the curve at depths 15,14,13,12,11,9.)
# Waits for VPN/DNS, submits, then VERIFIES registration in W&B.
set -uo pipefail
cd /Users/david.stanojevic/PycharmProjects/opaque
export OPAQUE_DOCKER_REGISTRY=europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml
export OPAQUE_DOCKER_TAG=david-stan-zenml-training-7e97389
export WANDB_MODE=online
PY=.zenml-client/bin/python
L=campaign_logs/margin_submit.log; : > "$L"

for i in $(seq 1 240); do   # up to ~12h
  if host zenml.labs.jb.gg >/dev/null 2>&1 && \
     [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://zenml.labs.jb.gg/api/v1/info)" = "200" ]; then
    echo "CONNECTED (attempt $i)"; break
  fi
  [ $((i % 20)) -eq 1 ] && echo "[wait $i/240] zenml unreachable (VPN down?)"
  sleep 180
done
[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://zenml.labs.jb.gg/api/v1/info)" = "200" ] \
  || { echo "GAVE UP: never reachable"; exit 1; }

for m in 0 4 6; do
  n="marg-nodp-a1-m${m}-s42"
  XSE_ADAPTIVE_DEPTH=1 XSE_ADAPTIVE_DEPTH_ALPHA=1 XSE_ADAPTIVE_DEPTH_MARGIN=$m \
    $PY deploy/zenml/run.py nodp --run-name "$n" --extra \
      --lora-xse-p-e 0.333 --lora-r 16 --num-epochs 1 --seed 42 >>"$L" 2>&1 \
    && echo "SUBMITTED $n (m=$m, expect depth ~$((15-m)))" || echo "SUBMIT-FAILED $n"
done

sleep 600
uv run python - <<'PYEOF' 2>/dev/null
import os
os.environ.setdefault("WANDB_BASE_URL","https://jetbrains.wandb.io")
import wandb
api=wandb.Api(timeout=60)
want={f"marg-nodp-a1-m{m}-s42" for m in (0,4,6)}
got={r.name for r in api.runs("federated-compute/opaque-lora-xs",
                              filters={"display_name":{"$regex":"^marg-nodp-"}})}
print(f"VERIFY registered={len(got & want)}/3")
if want-got: print("VERIFY-MISSING " + " ".join(sorted(want-got)))
PYEOF
echo "MARGIN SUBMIT DONE"
