#!/usr/bin/env bash
# Resilient submitter for the low-alpha non-DP sweep.
#
# Fixes the failure mode that has now bitten twice: a dropped VPN makes
# `run.py` fail with NameResolutionError after silent retries, and the exit code
# alone doesn't tell you whether the run actually reached the cluster.
#
#   1. wait for DNS + HTTPS reachability before touching anything
#   2. submit alpha in {0.05,0.1,0.15,0.2} (non-DP, m=2, seed 42)
#   3. VERIFY each run registered in W&B; report anything missing
set -uo pipefail
cd /Users/david.stanojevic/PycharmProjects/opaque
export OPAQUE_DOCKER_REGISTRY=europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml
export OPAQUE_DOCKER_TAG=david-stan-zenml-training-7e97389
export WANDB_MODE=online
PY=.zenml-client/bin/python
L=campaign_logs/lowalpha_submit2.log; : > "$L"
ALPHAS="0.05 0.1 0.15 0.2"

# --- 1. wait for connectivity (VPN) ---------------------------------------
for i in $(seq 1 240); do   # up to ~12h
  if host zenml.labs.jb.gg >/dev/null 2>&1 && \
     [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://zenml.labs.jb.gg/api/v1/info)" = "200" ]; then
    echo "CONNECTED (attempt $i) — submitting"; break
  fi
  [ $((i % 20)) -eq 1 ] && echo "[wait $i/240] zenml unreachable (VPN down?)"
  sleep 180
done
host zenml.labs.jb.gg >/dev/null 2>&1 || { echo "GAVE UP: never reachable"; exit 1; }

# --- 2. submit ------------------------------------------------------------
for a in $ALPHAS; do
  tag=$(echo "$a" | tr -d '.')
  n="lowa-nodp-a${tag}-m2-s42"
  XSE_ADAPTIVE_DEPTH=1 XSE_ADAPTIVE_DEPTH_ALPHA=$a XSE_ADAPTIVE_DEPTH_MARGIN=2 \
    $PY deploy/zenml/run.py nodp --run-name "$n" --extra \
      --lora-xse-p-e 0.333 --lora-r 16 --num-epochs 1 --seed 42 >>"$L" 2>&1 \
    && echo "SUBMITTED $n (alpha=$a)" || echo "SUBMIT-FAILED $n"
done

# --- 3. verify registration in W&B (not just the exit code) --------------
sleep 600
uv run python - <<'PYEOF' 2>/dev/null
import os
os.environ.setdefault("WANDB_BASE_URL","https://jetbrains.wandb.io")
import wandb
api=wandb.Api(timeout=60)
want={f"lowa-nodp-a{t}-m2-s42" for t in ("005","01","015","02")}
got={r.name for r in api.runs("federated-compute/opaque-lora-xs",
                              filters={"display_name":{"$regex":"^lowa-nodp-"}})}
print(f"VERIFY registered={len(got & want)}/4")
missing = want - got
if missing:
    print("VERIFY-MISSING " + " ".join(sorted(missing)))
PYEOF
echo "LOWALPHA SUBMIT DONE"
