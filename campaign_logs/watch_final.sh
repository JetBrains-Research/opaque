#!/usr/bin/env bash
# Wait for the 5 in-flight runs; when slots free, submit the 2 remaining s44 replicates.
set -uo pipefail
cd /Users/david.stanojevic/PycharmProjects/opaque
export OPAQUE_DOCKER_REGISTRY=europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml
export OPAQUE_DOCKER_TAG=david-stan-zenml-training-7e97389
export WANDB_MODE=online
export WANDB_BASE_URL=https://jetbrains.wandb.io
S44_DONE=0
for i in $(seq 1 300); do   # up to ~25h at 5min
  OUT=$(uv run python - <<'PY' 2>/dev/null
import os
os.environ.setdefault("WANDB_BASE_URL","https://jetbrains.wandb.io")
import wandb
api=wandb.Api(timeout=60)
want=["cmp-basexs-eps3-s42","cmp-basexs-eps3-s43","cmp-basexs-eps3-s44",
      "seedrep-ad-nodp-ainf-m1-s43","seedrep-ad-nodp-a2-m1-s43",
      "seedrep-ad-nodp-ainf-m1-s44","seedrep-ad-nodp-a2-m1-s44"]
by={}
for r in api.runs("federated-compute/opaque-lora-xs"):
    if r.name in want: by[r.name]=r
live=0; fin=0
for n in want:
    r=by.get(n)
    if r is None: continue
    if r.state in ("running","pending"): live+=1
    elif r.state=="finished": fin+=1
    st=r.state; ls=r.summary.get("eval/loss"); dp=r.summary.get("rotation/r_e_dyn")
    print(f"  {n:32s} {st:9s} step={r.summary.get('_step')} "
          f"loss={ls if ls is None else round(ls,5)} depth={dp if dp is None else round(dp,2)}")
print(f"LIVE={live} FIN={fin}")
PY
)
  echo "=== poll $i ==="; echo "$OUT"
  LIVE=$(echo "$OUT" | grep -o 'LIVE=[0-9]*' | cut -d= -f2)
  # once basexs+s43 wind down, launch the s44 pair
  if [ "$S44_DONE" = "0" ] && [ "${LIVE:-9}" -le 2 ]; then
    for spec in "inf:1:ainf" "2:1:a2"; do
      a="${spec%%:*}"; rest="${spec#*:}"; m="${rest%%:*}"; tag="${rest#*:}"
      n="seedrep-ad-nodp-${tag}-m${m}-s44"
      XSE_ADAPTIVE_DEPTH=1 XSE_ADAPTIVE_DEPTH_ALPHA=$a XSE_ADAPTIVE_DEPTH_MARGIN=$m \
        .zenml-client/bin/python deploy/zenml/run.py nodp --run-name "$n" --seed 44 \
          --extra --lora-xse-p-e 0.333 --lora-r 16 --num-epochs 1 \
        >>campaign_logs/seedrep_s44.log 2>&1 && echo "SUBMITTED $n" || echo "SUBMIT-FAIL $n"
    done
    S44_DONE=1
  fi
  if [ "$S44_DONE" = "1" ] && [ "${LIVE:-9}" = "0" ]; then echo "ALL DONE"; break; fi
  sleep 300
done
