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
#   1. full LoRA r=16          — the TARGET. Verified absent: at this operating
#                                point (7B, nz=0, r=16, lr 5e-2) lora_method is
#                                'lora-xs' in 62/62 runs, so a non-DP full-LoRA
#                                baseline has never been run. Three independent
#                                analyses reached this same n=0 conclusion.
#   2. LoRA-XSe d5, tau=1      — the current best-known config, and the matched
#                                CONTROL for every mechanism arm in batch 2.
#   3. frozen LoRA-XS, p_e=0   — no rotation. Also verified absent here: p_e==0
#                                in 0/62 runs. See the QUEUE note for the basis
#                                -normalization confound this exposes.
#
# All three: r=16, 2 epochs, seed 42, 520 steps, weight-decay 0, --eval-bpb.
#
# tau=1 rather than tau=2: they differ by 9.1e-6 (0.14x the 6.5e-5 floor, a
# statistical tie) so there is no accuracy reason to prefer either, but tau=1 is
# what the best on-disk run used and what the batch-2 mechanism tests need, and
# a shared control across batches is worth more than a marginal compute saving.
# Rotation cost is ~4ms against a ~15s step, so doubling rotation count is free.
#
# NOT in this batch, deliberately: the reset-in-place ablation, p_e=0 WITH
# --lora-xs-orthonormal-a, and the LoRA-SB gradient-basis init. All batch 2.
# This batch buys the denominators; batch 2 spends them.
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
#     https://zenml.labs.jb.gg/api/v1/info        # expect 200, not 000
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

# --weight-decay 0 is MANDATORY here, not a preference. Default is 0.01 and all
# 62 operating-point runs used it, but xse_sgd() (xse.py:901) takes no
# weight_decay parameter at all, while the non-XSe branch does pass it
# (train_causal_lm.py:1975). So at defaults, LoRA/frozen-XS train WITH decay and
# XSe trains WITHOUT it, and any "LoRA vs XSe" number is partly a weight-decay
# comparison. Setting 0 everywhere makes the arms actually matched. This
# confound is baked into every historical LoRA-vs-XSe contrast, including the
# eps=3 "rotation buys 103 floor units" figure.
COMMON="--lora-r 16 --num-epochs 2 --weight-decay 0 --eval-bpb"

# name | XSE env (or "-") | trainer --extra args
#
# Slot 3 is the plain p_e=0 frozen baseline, and it exists to expose a confound
# in the project's biggest claimed effect.
#
# core/svd.py:106-112 builds the frozen encoder as `U @ diag(S)` unless
# --lora-xs-orthonormal-a is set, so by default Sigma is ABSORBED into A and the
# basis is not an isometry. But xse.py:813-815 calls _normalize_layer at the
# FIRST ROTATION, which rescales A's rows and B's columns to unit length and
# absorbs the scale into R -- its own comment says "orthonormalize A/B (absorb
# Sigma into R)". _normalize_layer is reachable ONLY from xse_sgd, which
# train_causal_lm.py:1949-1953 constructs only when p_e > 0.
#
# So the conditioning of the basis is not matched across arms, and never has
# been: a ROTATING arm self-orthonormalizes after tau steps, while a FROZEN
# (p_e=0) arm and a full-LoRA arm keep Sigma in A for the entire run. Every
# "what rotation buys" number -- including the eps=3 figure of 103 floor units
# -- therefore compares a normalized basis against an unnormalized one, and
# some unknown part of it is basis normalization rather than rotation.
#
# This batch measures the confounded contrast as it has always been measured
# (slot 2 vs slot 3), so the new-scale number is directly interpretable against
# the historical one. Batch 2 then adds p_e=0 WITH --lora-xs-orthonormal-a,
# which is the same frozen arm with a matched basis, and the difference between
# that and slot 3 is the part of "rotation" that was actually normalization.
#
# Note for the record: applying --lora-xs-orthonormal-a to a ROTATING arm was
# considered for this slot and rejected after reading _normalize_layer. On a
# rotating arm the flag can only affect the first tau steps (tau=1 here) plus
# R's initial column scaling, because the first rotation normalizes the basis
# regardless. The flag matters where normalization never happens: p_e=0.
QUEUE=(
  "ref-lora-r16-s42|-|--lora-method lora --lora-xse-p-e 0 $COMMON"
  "ref-xse-d5t1-s42|-|--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1 $COMMON"
  "ref-xs-norot-s42|-|--lora-xse-p-e 0 $COMMON"
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
      https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
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
