# Running the LoRA-XSe campaign on a standalone GPU box

This is a self-contained runbook for reproducing and continuing the LoRA-XSe
experiments **without ZenML**, on any single machine with one NVIDIA GPU — a Coder
workspace, a JetTrain box, a bare VM. It also records the current results and the
open work, so a new machine (or a new person) can pick the campaign up from here.

The ZenML path is currently blocked by a cluster-side permissions regression
(§6), which is why this exists. Running the trainer directly is also *strictly
safer* — see §3.3, where the wrapper's argument handling has already invalidated
one batch.

---

## 1. What you need

| | |
|---|---|
| GPU | 1 × 80 GB (H100/A100-80G). Peak observed 77 GB at r=16, seq 1024, microbatch 16. See §3.4 for smaller cards. |
| CUDA | 12.x. The reference image is `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`. |
| Python | ≥ 3.11, < 3.13 |
| Disk | ~60 GB: ~25 GB venv (CUDA wheels), ~15 GB HF model cache, ~10 GB datasets |
| Network | GitHub (SSH, for the private submodule), Hugging Face, and `jetbrains.wandb.io` |
| Credentials | a W&B API key; an HF token; SSH access to `JetBrains-Research/LoRA-Privacy` |

`torch` is pinned `<2.11` in `pyproject.toml`. Do not lift that: torch ≥ 2.11 on
PyPI pulls `nvidia-*-cu13` wheels, which break `torch.cuda.is_available()` on a
CUDA 12 host.

---

## 2. Setup, from nothing

```bash
# 2.1 system packages — build-essential and libssl-dev are required, because
#     opaque-accounting is a Rust extension built by maturin at install time
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential pkg-config libssl-dev

# 2.2 uv and a Rust toolchain
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 2.3 the repo, WITH the submodule. vendor/lora-privacy is a separate private
#     repo over SSH; a plain clone leaves it empty and every import fails.
git clone --recurse-submodules git@github.com:JetBrains-Research/opaque.git
cd opaque
git checkout david-stan/zenml-training      # the campaign branch
git submodule update --init --recursive

# 2.4 dependencies. `--extra all` pulls the eval/dp extras the trainer needs.
uv sync --all-packages --extra all
uv pip install \
    "torchopt>=0.7.3" "datasets>=2.0.0" "transformers==4.57.1" \
    "peft>=0.18.0" "wandb>=0.16.0" "evalplus==0.3.1" ./vendor/lora-privacy
```

`zenml`, `gcsfs` and the GCP connectors from the container build are **not needed**
off-cluster. Skip them.

### 2.5 Verify before burning GPU time

```bash
.venv/bin/python -c "
import torch, transformers, peft, torchopt, datasets
import opaque.accounting
from opaque.functional import make_functional
from lora_privacy.peft_lora_xs import LoraXSConfig, xse_sgd
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print('transformers', transformers.__version__, '| peft', peft.__version__)
"
.venv/bin/python -m pytest examples/test_glue_data.py packages/opaque-core/tests/deploy -q
```

The pytest line is worth the 40 seconds. Among other things it checks the
optimizer-branch parity and env-passthrough invariants that have each silently
invalidated a batch (§3.3).

### 2.6 Environment

```bash
export WANDB_API_KEY=<key>
export WANDB_BASE_URL=https://jetbrains.wandb.io
export WANDB_ENTITY=federated-compute
export WANDB_PROJECT=opaque-lora-xs
export HF_TOKEN=<token>
export HF_HOME=/scratch/hf                 # keep the model cache off the root disk
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reclaims fragmented reserve
```

---

## 3. Running the trainer

### 3.1 The canonical pair

The whole campaign is a two-arm comparison. Everything else is a variation.

```bash
# ROTATING arm (LoRA-XSe)
.venv/bin/python examples/train_causal_lm.py \
  --preset qwen-coder-kstack-lora \
  --lora-method lora-xs --lora-r 16 --lora-alpha 16 \
  --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1 \
  --optimizer sgd --sgd-momentum 0.9 --learning-rate 5e-2 \
  --num-epochs 2 --weight-decay 0 \
  --noise-multiplier 0 \
  --eval-bpb --eval-batch-size 16 \
  --seed 42 --run-name rot-s42

# FROZEN arm (published LoRA-XS) — identical but for p_e
.venv/bin/python examples/train_causal_lm.py \
  ... --lora-xse-p-e 0 ... --run-name frz-s42
```

`p_e = 0.3125` at `r = 16` gives `r_e = floor(0.3125 × 16) = 5` explore directions
and `r_keep = 11`. **`r_e = floor(p_e · r)`**, so at r=8 the same p_e yields 2, an
achieved fraction of 0.25 rather than 0.3125.

### 3.2 `--noise-multiplier 0` is mandatory for a non-DP run

The `qwen-coder-kstack-lora` preset sets `target_epsilon = 3.0`. If you omit
`--noise-multiplier`, the trainer **calibrates a noise multiplier from that budget
and you get a DP run**. Under ZenML the `nodp` arm passed `--noise-multiplier 0`
for you; running directly, nothing does.

For a DP run, drop that flag and set the budget:

```bash
  --target-epsilon 3      # or 1
```

### 3.3 Three traps that have each already invalidated a batch

Running directly avoids the first, which is the worst of them. Read all three
before trusting a result.

1. **ZenML argv precedence (avoided here).** `deploy/zenml/run.py` prepends its
   own arm args, and `parse_args` skips presets for anything explicitly provided.
   So `--num-epochs`, `--lora-method`, `--lora-xse-p-e` and `--noise-multiplier`
   from the arm silently beat any preset. It made four GLUE runs train for **one
   epoch instead of twenty** while logging `num_epochs=1` as if intended. Direct
   invocation has no wrapper, so what you type is what runs.
2. **Optimizer-branch parity.** Fixed 2026-08-25, and pinned by
   `packages/opaque-core/tests/deploy/test_optimizer_branch_parity.py`. Before the
   fix the frozen branch built `sgd(lr, weight_decay)` and never passed momentum
   (`opaque.optimizers.sgd` defaults it to `0.0`), while the rotating branch used
   `momentum=args.sgd_momentum`. Every frozen-vs-rotating gap on the SGD path was
   plain SGD against heavy-ball — worth ~2.7e-3 of an 8.9e-3 headline. **It was
   invisible in W&B**, because the run config logs the argparse value, not what
   the optimizer received. The machine test is `xs/m_norm`: present ⇒ an active
   momentum buffer.
3. **Env knobs need an allowlist under ZenML only.** `deploy/zenml/settings.py`
   forwards `XSE_*` via an explicit list, and three ablations once ran with the
   knob silently dropped, comparing the default against itself to 0.0e-3. Running
   directly, the environment is inherited normally and this cannot happen.
   `packages/opaque-core/tests/deploy/test_env_passthrough.py` guards the ZenML path.

### 3.4 Memory, measured

| config | microbatch | seq | peak |
|---|---|---|---|
| Qwen2.5-Coder-7B, r=16 | 16 | 1024 | **77 GB** |
| Qwen2.5-Coder-7B, r=64 | 8 | 1024 | 47 GB |
| Mellum-4b, r=16 | 16 | 1024 | 49 GB |
| full LoRA r=16 | 8 | 1024 | 52 GB |
| RoBERTa-large / CoLA | 32 | 128 | **3 GB** |

On a 40 GB card use `--microbatch-size 4` and `--gradient-checkpointing`; gradient
accumulation keeps the global batch at the preset's 192 either way, so results stay
comparable. r=64 and full LoRA both **require** `--microbatch-size 8` — they OOM at
16 in the vocab-projection kernel. Always pin `--eval-batch-size` explicitly: it
defaults to `microbatch_size`, so changing one silently changes eval batching too.

**The GLUE/CoLA arm fits on almost anything (3 GB).** If GPU budget is tight, do
classification work first.

### 3.5 Other settings

```bash
# second model family
--preset mellum-kstack --sgd-momentum 0.9     # the preset does NOT set momentum,
                                              # and p_e>0 requires it (the rotation
                                              # SVDs the momentum buffer)
# GLUE classification
--preset roberta-large-glue --glue-task cola --num-epochs 50 --eval-steps 100 \
--clipping-mode fixed --clipping-norm 1e6 --learning-rate 1e-3 --classifier-lr 1e-2
```

Two notes on the GLUE path. `--clipping-mode fixed --clipping-norm 1e6` makes
AUTO-S clipping inert, which is required for the learning rate to mean what it
means in the LoRA-XS paper; verified to reproduce the exact mean per-example
gradient. And `--classifier-lr` exists because the paper tunes the head's rate
separately (1e-2 against an adapter 1e-3 on CoLA) — without it the randomly
initialised head dominates the gradient (mean norm 95.6 versus 0.30 on the
causal-LM task).

### 3.6 Downstream metrics

```bash
--output-dir /scratch/<run-name> --eval-humaneval --eval-humaneval-n-samples 164 --eval-mbpp
```

**`--output-dir` is mandatory** — the entire save + downstream + parameter
write-back block is gated on it.

Every `downstream/*` number logged before 2026-09-08 is void: training happens in
functional form and the trained core and rotated basis were never written back into
the module, so HumanEval/MBPP scored the model at initialisation (ΔW ≈ 0) —
identically for every arm, across roughly 225 runs. Fixed by `_write_back`. **The
fix has never been exercised on a GPU**, so the first downstream numbers you get are
also the test of the fix: confirm they differ from the base model at all before
trusting any comparison.

### 3.7 Env-gated ablations

```bash
XSE_RESET_IN_PLACE=1   # re-coordinatise + wipe, span PINNED
XSE_ZERO_EXPLORE=1     # span moves, explore band zeroed
XSE_RENORM=1           # preserve ||R|| across a rotation
XSE_KEEP_SOURCE=core   # select kept directions from SVD(R) instead of SVD(M)
XSE_ADAM_PRECOND=scalar
XSE_ADAM_STATE=carry   # leave nu in the stale basis (GaLore's policy)
XSE_MP_SHRINKAGE=<c>   # Marchenko-Pastur shrinkage of the momentum spectrum
```

---

## 4. Current results

Qwen2.5-Coder-7B on KStack, final eval loss, **momentum-correct baselines only**.
Every adapter arm trains 50,176 parameters; full LoRA r=16 trains 40,370,176.

| comparison | rotation | frozen | gap |
|---|---|---|---|
| Qwen, SGD, 520 steps | 0.693218 ± 6.3e-5 (n=3) | 0.697987 ± 1.4e-4 (n=3) | **4.77e-3**, Welch t = 54, p = 3.1e-5 |
| Qwen, AdamW | 0.691887 ± 2.8e-6 (n=2) | 0.695249 ± 1.7e-4 (n=2) | **3.36e-3** |
| **Mellum-4b, SGD** | 0.512284 ± 1.3e-4 (n=2) | 0.515104 ± 4.2e-4 (n=2) | **2.82e-3** |

The AdamW pair is momentum-clean *by construction* — both branches pass
`betas=(sgd_momentum, adam_beta2)` — and has never needed revision. It is the
figure to quote if only one is quoted.

### 4.1 The headline moved twice, both times downward

| | reference | why it was wrong |
|---|---|---|
| 8.86e-3 | frozen at momentum 0.0 | the frozen branch never received momentum |
| 6.19e-3 | reset-in-place | wrong reference: the wipe is itself harmful |
| **4.77e-3** | frozen + momentum | correct baseline |

### 4.2 The decomposition, reproduced on both models

| | wipe costs | relocation gains | net |
|---|---|---|---|
| Qwen | −1.42e-3 | +6.19e-3 | **+4.77e-3** |
| Mellum | −0.95e-3 | +3.77e-3 | **+2.82e-3** |

Re-coordinatising and truncating is **harmful** on both models; moving the span more
than pays for it. `XSE_ZERO_EXPLORE` (span moves, band zeroed) tracks full rotation
to 2.2e-5 at n=2, so the retained explore residue contributes nothing — the whole
effect is the span change.

### 4.3 The mechanism

**LoRA-XS pins the adapter to the top-r singular subspaces of W₀ — the *pretrained*
weight — which have no relation to the fine-tuning gradient. The rotation is
randomized two-sided subspace iteration (a block power method) on the momentum that
repairs that mis-specification, and the repair is a one-time transient.**

`xs/grad_explore_frac` is the share of core-gradient energy landing on the 5 freshly
drawn *random* directions. Random directions in 3584 dimensions should catch almost
nothing; a large value means the kept span is badly placed.

| task | starts | floor | collapse | outcome |
|---|---|---|---|---|
| Qwen / KStack | 0.518 | 0.060 | 8.7× | +4.77e-3 |
| Mellum / KStack | 0.276 | 0.088 | 3.2× | +2.82e-3 |
| CoLA | 0.390 | 0.349 | **1.1×** | **loses** |

Monotone across all three, including the sign flip. Ordering is correct;
**scaling is sublinear** (deficit ratio 2.43 against gap ratio 1.69) — do not claim
proportionality.

Supporting evidence: the alignment floor is reached at step 48/35/28/21 for
r_e = 2/5/10/20, a log-log slope of −0.355 (a Monte-Carlo oversampling law, the
signature of randomized subspace iteration); the loss gap peaks at step 80/100/70/50,
**always after** that rank's floor; and `XSE_KEEP_SOURCE=core`, which never realigns,
returns exactly to baseline.

**It is a transient, not a rate.** The gap peaks at 1.37e-2 by step 100 of 1560 and
then *erodes* to 7.9e-3. After step 100 rotation descends 1.8–2.5× **slower** than
frozen. The step-dilation the frozen arm needs to match rotation is 5× at rotation's
step 30, 24× at step 60, and **unbounded past step 70** — it never arrives in 1560
steps. A diverging dilation cannot be a time rescaling, which rules out every
effective-learning-rate account.

Two caveats to carry: `grad_explore_frac` is **partly definitional** (the kept frames
*are* M's top directions), and the mechanism has **no damage term**, so it does not
explain the CoLA failure. Novelty is also limited — this is GaLore's stated rationale
for subspace switching, quantified.

### 4.4 Rank

Flat at constant effective learning rate. With `α = 16` fixed the gap appears to grow
monotonically (7.93 → 8.86 → 10.60 → 11.24 e-3 at r = 8/16/32/64, reproduced on two
seeds), but `s = α/r`, so effective LR falls as 1/r² and the r=64 arm trains at 1/16th
the rate. With `α = r`:

| r | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| gap | 8.39e-3 | 8.84e-3 | 8.96e-3 | 8.00e-3 |

**Always set `--lora-alpha` equal to `--lora-r` when sweeping rank.** The apparent
rank law is retracted; the effect is instead *stable across an 8× rank range*, which
is arguably the better claim.

### 4.5 Where it fails

GLUE CoLA, RoBERTa-large, Matthews (higher better), paper reference 0.670:

| arm | Matthews |
|---|---|
| frozen r=16 | **0.6348 ± 0.0076** (n=3) |
| frozen r=11 | 0.6091 ± 0.0098 (n=3) |
| rotation τ=10 | 0.5513 |
| rotation τ=50 | 0.5440 |
| rotation τ=1 | 0.4343 |

Deficit **0.0835**. With warmup 2670 the adapter reaches 0.6055 *during warmup* and
Matthews collapses to **0.0000** at the first rotation step. So rotation does not
merely fail to help a converged classification adapter — it destroys one.

The frozen arm reproduces the literature (0.6348 against 0.670), so the harness is
calibrated and the failure is real rather than instrumental.

**Four mechanisms eliminated by dedicated measurement**, not set aside:

| candidate | how it died |
|---|---|
| momentum non-concentration | CoLA's momentum is *more* concentrated (eff. rank 1.22 vs 1.94) |
| basis preconditioning | the orthonormal control made frozen **7.4e-3 worse** |
| reachability | frozen r=11 vs r=16 costs 1.9 sd; rotation sits 5.9 sd below frozen r=11 |
| head non-stationarity | the head *is* settled by step 2670 and rotation still zeroes Matthews |

### 4.6 Every principled intervention lost

| intervention | Δ vs baseline | |
|---|---|---|
| `XSE_KEEP_SOURCE=core` (Eckart–Young optimal retention) | +1.01e-2 | collapses to frozen |
| `XSE_ADAM_PRECOND=scalar` (the unique equivariant diagonal choice) | +1.68e-3 | worse |
| `XSE_RENORM=1` (remove the implicit weight decay) | +1.18e-3 | worse |
| `XSE_ADAM_STATE=carry` | −8.6e-5 | genuine null |
| `XSE_MP_SHRINKAGE` | +8e-6 (0.08 sd) | null |

Legible in hindsight: `keep=core` and `renorm` either **reduce movement** or **remove
a useful side effect**, and the other two optimize quantities that do not matter. The
theory kept refining the parts of the algorithm that were not doing the work.

Also retracted: **"beats full LoRA at 805× fewer parameters."** The comparison rested
on a baseline that had diverged, and the quoted figure was a value the model passed
through at step 10 on its way up. Against a properly tuned baseline it does not hold.

---

## 5. Future steps

Ranked by information per GPU-hour.

### 5.1 Immediate — three runs, closes the record

```bash
# converged momentum-correct baseline; the 520-step number is settled, this is 1560
for s in 42 43 44; do
  .venv/bin/python examples/train_causal_lm.py --preset qwen-coder-kstack-lora \
    --lora-method lora-xs --lora-r 16 --lora-alpha 16 --lora-xse-p-e 0 \
    --optimizer sgd --sgd-momentum 0.9 --learning-rate 5e-2 \
    --num-epochs 6 --weight-decay 0 --noise-multiplier 0 \
    --eval-bpb --eval-batch-size 16 --seed $s --run-name q-norot-mom-e6b-s$s
done
```

Expect ~4.2e-3 if the ~89% retention the confounded version showed carries over.

### 5.2 Downstream metrics — 2 runs, and they test their own fix

Run the canonical pair from §3.1 with the §3.6 flags added. **Read them as a test of
`_write_back` first and a result second**: confirm the scores move off the base model
at all, and that the two arms differ. If both are flat, downstream on this task is
uninformative at these ΔW magnitudes and should be dropped rather than paid for.

### 5.3 DP-SGD at ε = 3 and ε = 1 — 8 runs, and a real test of the mechanism

Drop `--noise-multiplier 0`, add `--target-epsilon 3` (or `1`), two seeds per arm.

This is more than another setting. The mechanism is a **power method run on the
momentum buffer**, and DP clips and noises the gradient — injecting noise into
exactly the signal the rotation reads to choose directions.

> **Pre-registered prediction: gap at ε=1 < gap at ε=3 < gap at no-DP (4.77e-3).**
> If the gap is *unchanged* under noise, the realignment mechanism is wrong.

State this before looking. It is the strongest falsification test available, and it
costs 8 runs.

### 5.4 Lower value, listed so it is not rediscovered

- **Seeds on the α=r sweep** (n=1 per cell). Cheap, but the endpoints already agree
  with the α=16 sweep at n=2.
- **Gradient-basis init** (`--lora-xs-init grad`). Implemented, crashed twice
  (tuple-arity, then OOM), rewritten, **never successfully run**. Low priority: the
  one-step optimality claim is published four times over, including by LoRA-XS's own
  Theorem 1.
- **A third task.** The mechanism predicts the outcome from the alignment deficit,
  so a task whose deficit is measured *first* and whose outcome is then predicted
  would be the strongest confirmation available. The deficit is readable from
  `xs/grad_explore_frac` within ~50 steps, so this is cheap to screen.

### 5.5 Do not spend GPU time on

The second-moment transport work (small, and it errs both ways), error feedback
(published, and the naive route is published as blocked), the confinement
nonconvergence theorem (correct, sharp, and empirically inert — a dedicated
experiment showed the quantity it bounds costs almost nothing in loss on either
task), and any further attempt to rescue CoLA with τ, p_e or warmup (all three tried
and all three lose).

---

## 6. The ZenML blocker, for the record

Every submission since ~2026-08-31 fails at the door:

```
PipelineSubmissionError: Orchestrator failed to submit Kubernetes job with
'Forbidden' (403). rolebindings.rbac.authorization.k8s.io is forbidden: User
"system:serviceaccount:zenml-workload-common-gpus:zenml-orchestrator" cannot
create resource "rolebindings" in API group "rbac.authorization.k8s.io" in the
namespace "zenml"
```

The orchestrator service account lost permission to create `rolebindings` in the
`zenml` namespace. **This is a regression** — batches 26–28 ran on the identical code
path in late August — and it is not fixable from the client side. It also affects
anything else using that stack, not just this project. The message names the exact
principal, verb, resource, API group and namespace, so it should be quick for whoever
administers `gke-europe-west4`.

Nothing was lost to it: the jobs were rejected before starting, so no partial or
misconfigured runs exist.

### 6.1 When it is restored

The queue scripts in `campaign_logs/queue/` still work unmodified. `batch29` (the
converged baseline) and `batch30` (downstream + DP) are the two outstanding ones. Each
polls Artifact Registry for its pinned image, gates on a concurrency cap, and skips
runs whose W&B name already exists, so they are safe to re-run as-is.

Images already built and verified present: `david-stan-zenml-training-dfb3f72`
(momentum fix) and `david-stan-zenml-training-8e8b021` (write-back fix, needed for
downstream metrics).

---

## 7. Reading results safely

Six defects in this campaign produced plausible-looking numbers that were wrong.
Every one is now guarded by a test or a diagnostic, but the habits are worth keeping.

- **Quote final-step values, never `eval/loss_min`.** One retracted claim came from a
  `loss_min` at step 10 of a run that then diverged to 1.47.
- **Check `state` and `_step` before quoting anything.** Mid-flight over-reads
  produced four wrong numbers, including a CoLA arm reported at 0.2032 whose final
  maximum was 0.4729.
- **Compare at matched steps**, not final against mid-flight. Doing this changed one
  conclusion outright: an arm that looked 6e-4 apart was 3e-5 apart.
- **`xs/m_norm` present ⇒ momentum was active.** The run config cannot tell you this;
  it logs the argparse value, not what the optimizer received.
- **Bit-identical results across an ablation mean the knob never applied.** That is
  how the env-passthrough defect surfaced.
- **A run finishing at a suspiciously round small step count** is the ZenML
  argv-precedence trap.
- Seed sd is ~6e-5 on the Qwen causal-LM family. Treat anything under 1.5e-4 as
  indistinguishable.

Method derivations and theory live in `vendor/lora-privacy/docs/` — in particular
`lora-xse-explained.md`, `lora-xse-claim-audit-2026-08-22.md` (novelty verdicts and
corrections) and `status-2026-08-24.md`.
