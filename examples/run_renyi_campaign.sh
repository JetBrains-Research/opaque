#!/usr/bin/env bash
# ==============================================================================
# Rényi / LoRA-XSe empirical campaign — full non-DP vs DP comparison matrix.
#
# Answers "does this pay off at all?" by sweeping every comparison that is
# runnable with the CURRENT code (examples/train_causal_lm.py):
#   - method family: LoRA, LoRA-XS (uniform), LoRA-XSe (rotation), LoRA-XSe +
#     adaptive depth over the alpha x margin grid;
#   - privacy: non-DP (--noise-multiplier 0) and DP at eps in {1,3,8};
#   - regime axis: rank r in {8,16,32};
#   - rotation cadence: interval in {1,5,10};
#   - seeds for the load-bearing cells.
#
# NOT covered here (need code first — separate phase, see
# docs/renyi-utility-experiments-plan.md): AdaLoRA baseline and the
# Rényi/stable-rank PER-LAYER allocation method. This script benchmarks the
# LoRA-XSe *family* that already exists; it tells you whether the direction is
# worth building those on.
#
# USAGE
#   bash examples/run_renyi_campaign.sh                 # run tiers A B C D in order
#   TIERS="A" bash examples/run_renyi_campaign.sh       # just the core head-to-head
#   DRY_RUN=1 bash examples/run_renyi_campaign.sh       # print the plan, run nothing
#   FAST=1 bash examples/run_renyi_campaign.sh          # quick signal (1 epoch, 20k
#                                                       #   samples, eval-loss only)
#   EVAL_DOWNSTREAM=1 bash examples/run_renyi_campaign.sh# add HumanEval+/MBPP+
#
# Each line is an INDEPENDENT run. On one GPU they run sequentially (see the
# count/time estimate printed at the end). To parallelize, pipe the DRY_RUN
# output to your scheduler (one job per line) instead of running this directly.
# ==============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

PRESET="qwen-coder-kstack-lora"          # Qwen2.5-Coder-7B, KStack, lr5e-2, mom0.9
PE="0.333"                                # XSe exploration fraction (activates rotation)

EPOCHS="${EPOCHS:-1}"
SAMPLES="${SAMPLES:-}"                     # empty = preset default (50000)
SEEDS="${SEEDS:-42 43}"
TIERS="${TIERS:-A B C D}"
DRY_RUN="${DRY_RUN:-0}"
FAST="${FAST:-0}"
EVAL_DOWNSTREAM="${EVAL_DOWNSTREAM:-0}"
LOG_DIR="${LOG_DIR:-campaign_logs}"

if [[ "$FAST" == "1" ]]; then EPOCHS=1; SAMPLES="${SAMPLES:-20000}"; EVAL_DOWNSTREAM=0; fi
mkdir -p "$LOG_DIR"

COMMON=( --preset "$PRESET" --num-epochs "$EPOCHS" )
[[ -n "$SAMPLES" ]] && COMMON+=( --num-train-samples "$SAMPLES" )
[[ "$EVAL_DOWNSTREAM" == "1" ]] && COMMON+=( --eval-humaneval --eval-mbpp )

COUNT=0
# run <name> <dp_flag...> -- <method_flag...> [with ENV k=v k=v]
# We pass adaptive-depth via env (XSE_ADAPTIVE_DEPTH / _ALPHA / _MARGIN).
run () {
  # NB: everything here MUST be `local` — callers use loop vars named a/m/s/r.
  local name="$1"; shift
  # split args at the literal "ENV"
  local args=() envs=() seen_env=0 tok desc
  for tok in "$@"; do
    if [[ "$tok" == "ENV" ]]; then seen_env=1; continue; fi
    if [[ "$seen_env" == "0" ]]; then args+=("$tok"); else envs+=("$tok"); fi
  done
  COUNT=$((COUNT+1))
  local desc="[$COUNT] RUN_NAME=$name  ${envs[*]:-}  ${args[*]}"
  echo "$desc"
  if [[ "$DRY_RUN" == "1" ]]; then return; fi
  RUN_NAME="$name" env "${envs[@]}" \
    uv run python examples/train_causal_lm.py "${COMMON[@]}" "${args[@]}" \
    > "$LOG_DIR/$name.log" 2>&1 || echo "  !! $name FAILED (see $LOG_DIR/$name.log)"
}

# Method definitions (args after "--") ---------------------------------------
# M1 LoRA            M2 LoRA-XS uniform     M3 LoRA-XSe rotation-only
# M4-M7 LoRA-XSe + adaptive depth at alpha in {0.5,1,2,inf}, margin m.
lora ()      { echo "--lora-method lora"; }
loraxs ()    { echo "--lora-method lora-xs --lora-xse-p-e 0.0"; }           # uniform XS, no rotation
loraxse ()   { echo "--lora-method lora-xs --lora-xse-p-e $PE"; }          # rotation, adaptive OFF
# adaptive-depth env for a given alpha/margin:
adenv ()     { echo "XSE_ADAPTIVE_DEPTH=1 XSE_ADAPTIVE_DEPTH_ALPHA=$1 XSE_ADAPTIVE_DEPTH_MARGIN=$2"; }

dp_nodp ()   { echo "--noise-multiplier 0"; }
dp_eps ()    { echo "--target-epsilon $1"; }

# ------------------------------------------------------------------ TIER A ---
# Core head-to-head: does LoRA-XSe beat LoRA-XS and tie/beat LoRA, DP & non-DP?
# 4 methods x {non-DP, DP eps3} x seeds. r=16 (tuned).
if [[ " $TIERS " == *" A "* ]]; then
  echo "### TIER A — core head-to-head (r=16, non-DP & eps3, seeds: $SEEDS)"
  for s in $SEEDS; do
    for dp in "nodp" "eps3"; do
      DPFLAG=$([[ "$dp" == "nodp" ]] && dp_nodp || dp_eps 3)
      run "camp-A-lora-$dp-r16-s$s"        $DPFLAG --lora-r 16 --seed $s $(lora)     ENV
      run "camp-A-loraxs-$dp-r16-s$s"      $DPFLAG --lora-r 16 --seed $s $(loraxs)   ENV
      run "camp-A-xse-rot-$dp-r16-s$s"     $DPFLAG --lora-r 16 --seed $s $(loraxse)  ENV
      run "camp-A-xse-aInf-m2-$dp-r16-s$s" $DPFLAG --lora-r 16 --seed $s $(loraxse)  ENV $(adenv inf 2)
    done
  done
fi

# ------------------------------------------------------------------ TIER B ---
# alpha x margin grid for adaptive depth (DP eps3, r=16, seed 42). Which (alpha,m)
# is best under DP? Prediction: high alpha, m~2.
if [[ " $TIERS " == *" B "* ]]; then
  echo "### TIER B — alpha x margin grid (DP eps3, r=16, seed 42)"
  for a in 0.5 1 2 inf; do
    for m in 1 2 3; do
      run "camp-B-a${a}-m${m}-eps3-r16-s42" $(dp_eps 3) --lora-r 16 --seed 42 $(loraxse) ENV $(adenv $a $m)
    done
  done
fi

# ------------------------------------------------------------------ TIER C ---
# Regime map: adaptive alpha=inf m=2 across (privacy) x (rank). Predicts the win
# concentrates in high-noise (low eps) / small-r corner.
if [[ " $TIERS " == *" C "* ]]; then
  echo "### TIER C — regime map (alpha=inf, m=2, seed 42): {non-DP,eps8,eps3,eps1} x r{8,16,32}"
  for r in 8 16 32; do
    for dp in "nodp" "eps8" "eps3" "eps1"; do
      case "$dp" in
        nodp) DPFLAG=$(dp_nodp);; eps8) DPFLAG=$(dp_eps 8);;
        eps3) DPFLAG=$(dp_eps 3);; eps1) DPFLAG=$(dp_eps 1);;
      esac
      run "camp-C-aInf-m2-$dp-r$r-s42" $DPFLAG --lora-r $r --seed 42 $(loraxse) ENV $(adenv inf 2)
    done
  done
fi

# ------------------------------------------------------------------ TIER D ---
# Rotation cadence (H4): does rotating every step hurt more under DP than non-DP?
if [[ " $TIERS " == *" D "* ]]; then
  echo "### TIER D — rotation cadence (alpha=inf m=2, r=16, seed 42): interval{1,5,10} x {non-DP,eps3}"
  for iv in 1 5 10; do
    for dp in "nodp" "eps3"; do
      DPFLAG=$([[ "$dp" == "nodp" ]] && dp_nodp || dp_eps 3)
      run "camp-D-int${iv}-$dp-r16-s42" $DPFLAG --lora-r 16 --seed 42 \
          --lora-xse-rotation-step-interval $iv $(loraxse) ENV $(adenv inf 2)
    done
  done
fi

echo
echo "Planned $COUNT runs. At ~1-1.5h/run (Qwen-7B, ${EPOCHS} epoch) that is"
echo "~$((COUNT)) - $((COUNT*3/2)) GPU-hours sequential; parallelize across GPUs to fit a night."
echo "Logs: $LOG_DIR/  |  metrics: wandb federated-compute/opaque-lora-xs (rotation/r_eff_*, eval/loss_min)."
[[ "$DRY_RUN" == "1" ]] && echo "(DRY_RUN — nothing was executed.)"
