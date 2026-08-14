#!/usr/bin/env bash
set -uo pipefail
cd /Users/david.stanojevic/PycharmProjects/opaque
export OPAQUE_DOCKER_REGISTRY=europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml
export OPAQUE_DOCKER_TAG=david-stan-zenml-training-7e97389
export WANDB_MODE=online WANDB_BASE_URL=https://jetbrains.wandb.io
S44=0
for i in $(seq 1 400); do
  OUT=$(uv run python - <<'PY' 2>/dev/null
import os
os.environ.setdefault("WANDB_BASE_URL","https://jetbrains.wandb.io")
import wandb
want=["cmp-basexs-eps3-s42","cmp-basexs-eps3-s43","cmp-basexs-eps3-s44",
      "seedrep-ad-nodp-ainf-m1-s43","seedrep-ad-nodp-a2-m1-s43",
      "seedrep-ad-nodp-ainf-m1-s44","seedrep-ad-nodp-a2-m1-s44"]
try:
    api=wandb.Api(timeout=60)
    by={r.name:r for r in api.runs("federated-compute/opaque-lora-xs") if r.name in want}
except Exception as e:
    print("POLL-ERR"); raise SystemExit(0)
live=term=0
for n in want:
    r=by.get(n)
    if r is None: print(f"  {n:32s} -absent-"); continue
    if r.state in ("running","pending"): live+=1
    else: term+=1
    l=r.summary.get("eval/loss"); d=r.summary.get("rotation/r_e_dyn")
    print(f"  {n:32s} {r.state:9s} step={r.summary.get('_step')} "
          f"loss={None if l is None else round(l,5)} depth={None if d is None else round(d,2)}")
print(f"LIVE={live} TERM={term}")
PY
)
  echo "=== poll $i ==="; echo "$OUT"
  LIVE=$(echo "$OUT" | grep -o 'LIVE=[0-9]*' | cut -d= -f2 || echo 9)
  TERM=$(echo "$OUT" | grep -o 'TERM=[0-9]*' | cut -d= -f2 || echo 0)
  if [ "$S44" = "0" ] && [ -n "${LIVE:-}" ] && [ "${LIVE:-9}" -le 2 ]; then
    for spec in "inf:ainf" "2:a2"; do
      a="${spec%%:*}"; tag="${spec#*:}"; n="seedrep-ad-nodp-${tag}-m1-s44"
      XSE_ADAPTIVE_DEPTH=1 XSE_ADAPTIVE_DEPTH_ALPHA=$a XSE_ADAPTIVE_DEPTH_MARGIN=1 \
        .zenml-client/bin/python deploy/zenml/run.py nodp --run-name "$n" --seed 44 \
          --extra --lora-xse-p-e 0.333 --lora-r 16 --num-epochs 1 \
        >>campaign_logs/seedrep_s44.log 2>&1 && echo "SUBMITTED $n" || echo "SUBMIT-FAIL $n"
    done
    S44=1; sleep 240
  fi
  # exit only when all 7 exist AND none are live (covers finished/failed/crashed/killed)
  if [ "$S44" = "1" ] && [ "${LIVE:-9}" = "0" ] && [ "${TERM:-0}" = "7" ]; then
    echo "ALL 7 TERMINAL"; break
  fi
  sleep 300
done
echo "WAITER EXIT"
