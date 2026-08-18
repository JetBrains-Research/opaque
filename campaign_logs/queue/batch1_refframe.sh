#!/usr/bin/env bash
# BATCH 1 — the reference frame on the NEW image.
#
# Why this batch and not the reset-in-place ablation first: the padding-mask fix
# (image ...-fec7ae2) roughly doubles absolute eval loss, because ~49.5% of
# scored positions used to be <eos>-after-<eos>. The fix is SNR-neutral, so no
# past verdict changes, but NO number from this image is comparable to any of
# the 297 historical runs. Until the three reference points below exist on the
# new scale, the sentence "LoRA-XSe matches/beats LoRA" cannot be evaluated at
# all. Everything downstream is measured against these three.
#
#   1. full LoRA r=16      — the target. This is the preset's own tuned config
#                            (comment at train_causal_lm.py:1035 records its
#                            pre-fix best as eval 0.3449 @ step 520).
#   2. LoRA-XS, p_e=0      — no rotation at all. Isolates what rotation buys.
#   3. LoRA-XSe d5, tau=2  — the current best-known config. Doubles as the
#                            matched CONTROL for the reset-in-place arm in
#                            batch 2, so that comparison stays internal.
#
# All three: r=16, 2 epochs, seed 42, 520 steps, --eval-bpb.
#
# tau=2 because the tau sweep saturates there: tau=1 vs tau=2 differed 9.1e-6
# (0.14x the 6.5e-5 floor -- a tie) while tau=1 vs tau=10 spanned 27x the floor.
# There is no reason to pay for tau=1.
#
# --eval-bpb is on because bits-per-byte carries ~20x the SNR of aggregate eval
# loss on code (arXiv 2508.13144) and yields PER-EXAMPLE values, which allows a
# paired bootstrap instead of comparing two scalars across a 6.5e-5 floor. This
# is the first image that can actually compute it.
#
# DEPTH ARITHMETIC -- the easy way to silently run the wrong experiment:
#   r_e = p_e * r, and r is 16 here (preset), NOT 32.
#   depth 5  <=>  p_e 0.3125   (0.3125 * 16 = 5.0)
#   p_e 0.15625 would give 2.5 -> depth 2, a different experiment entirely.
#
# Usage (VPN MUST be up -- check FIRST, a 000 means submissions silently die):
#   curl -s -o /dev/null -w '%{http_code}\n' --max-time 10 \
#     https://zenml.grazie.aws.intellij.net        # expect 200, not 000
#   ./campaign_logs/queue/batch1_refframe.sh
#
# Idempotent: a run whose name already exists in W&B is skipped, so re-running
# after a VPN drop resumes instead of duplicating.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
# The image built from fec7ae24: padding fix + LR schedule + reset-in-place +
# rotation/sv0..sv7 + eval-bpb. Do not point this at ...-7e97389; that one has
# none of it.
export OPAQUE_DOCKER_TAG="${OPAQUE_DOCKER_TAG:-david-stan-zenml-training-fec7ae2}"
export WANDB_MODE=online
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"

# 3, not 4: the user asked to keep the cluster uncluttered, and over-subscribing
# the ~5-slot quota co-located pods and crashed 7 runs in July.
MAXCONC="${MAXCONC:-3}"
LOGDIR=campaign_logs/queue
mkdir -p "$LOGDIR"

COMMON="--lora-r 16 --num-epochs 2 --eval-bpb"

# name | XSE env (or "-") | trainer --extra args
QUEUE=(
  "ref-lora-r16-s42|-|--lora-method lora --lora-xse-p-e 0 $COMMON"
  "ref-xs-norot-s42|-|--lora-xse-p-e 0 $COMMON"
  "ref-xse-d5t2-s42|-|--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 2 $COMMON"
)

# A fresh wandb.Api() per call, deliberately. Reusing one Api object caches run
# objects, which once made a waiter report step=3 for five hours.
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

# Fail fast rather than submitting into a void: with the VPN down, ZenML returns
# 000 and run.py dies partway through "Archiving pipeline code directory".
ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
      https://zenml.grazie.aws.intellij.net 2>/dev/null)"
if [[ "$ZC" != "200" && "$ZC" != "302" && "$ZC" != "401" ]]; then
  echo "[ABORT] ZenML unreachable (HTTP $ZC) — connect the VPN and re-run."
  exit 1
fi
echo "[ok] ZenML reachable (HTTP $ZC); image tag $OPAQUE_DOCKER_TAG"

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
    .zenml-client/bin/python deploy/zenml/run.py nodp --run-name "$NAME" --seed 42 \
      --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  else
    env $ENVS .zenml-client/bin/python deploy/zenml/run.py nodp --run-name "$NAME" --seed 42 \
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
echo "BATCH 1 DONE"
