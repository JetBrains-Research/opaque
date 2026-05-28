# Feature Validation Agent

Autonomous multi-hour agent that validates code changes by running GPU
experiments on Cadence and analyzing results from W&B. Claude Code only
(requires `/loop` for multi-hour polling; Junie cannot do this).

## Invocation

```
/agents feature-validation
```

Or invoke manually when a branch has changes worth validating on GPU.

## Workflow

### Phase 1: Understand the change

1. Run `git diff main...HEAD` to identify what changed.
2. Read the modified files to understand the feature's claims — what it
   should improve, what invariants it must preserve.
3. Identify which training entry point and Cadence preset to use:
   - DP-SGD changes → `train_causal_lm.py` with preset
     `.cadence/configs/train_causal_lm (mellum_kstack).yaml`
   - DP-FTRL changes → `train_dp_ftrl.py` with preset
     `.cadence/configs/train_dp_ftrl (mellum_kstack).yaml`
   - Performance/kernel changes → both, comparing wall-clock time
   - Distributed changes → use the `_distributed` preset variants

### Phase 2: Design experiments

Design a validation plan with at least:

- **Baseline run**: current `main` behavior (or a no-change control)
- **Variant run**: the branch under test
- **What to measure**: specific W&B metrics that the change should
  affect, plus invariants that must hold

Present the plan to the user and get approval **once**. After approval,
run fully autonomously — no further user interaction until the final
report.

### Phase 3: Submit runs

Use the `jbr-fed-researcher:cadence-experiments` skill to submit runs.

Naming convention for W&B runs:
```
val/<branch-slug>/<role>
```
where `<role>` is `baseline`, `variant`, `variant-2`, etc.

Example submission via skill:
```
/cadence-experiments submit
  --preset ".cadence/configs/train_causal_lm (mellum_kstack).yaml"
  -e RUN_NAME="val/my-feature/baseline"
  -e EXTRA_ARGS="--max-steps 200"
```

Always set `--max-steps` to keep runs short enough to validate
(100–500 steps is typical; longer only if the effect needs warmup).

### Phase 4: Monitor

Use `/loop 20m` to poll W&B for run status every 20 minutes.

Use the `jbr-fed-researcher:wandb-metrics` skill to check:
- Run state (running / finished / crashed)
- Key metrics: `train/loss`, `train/grad_norm`, `eval/loss`,
  `privacy/epsilon`, `perf/step_time_ms`

Continue polling until all runs in the current round finish. Expect
runs to take 30 minutes to 3+ hours depending on step count and model
size.

If a run crashes, read the Cadence logs to diagnose. If it's an
infrastructure issue (OOM, node failure), resubmit. If it's a code
bug, report immediately — do not continue.

### Phase 5: Analyze

Compare baseline vs variant using **scientific plausibility checks**,
not fixed thresholds. The checks depend on what the feature claims:

#### Always check (DP invariants)

- `privacy/epsilon` must be finite and must match the accountant's
  prediction to within floating-point tolerance. If the variant's
  epsilon diverges from baseline at the same noise multiplier and
  sample rate, something is broken.
- `train/grad_norm` post-clipping must respect the clipping bound.
  Norms consistently above the bound indicate a clipping bug.
- Loss must not be NaN or Inf at any logged step.

#### Performance claims

- If the feature claims faster training: variant `perf/step_time_ms`
  should be measurably lower. A 1–2% difference is noise; expect 5%+
  for a real improvement. Check that `train/loss` curves are
  comparable — faster is meaningless if the model diverges.

#### Quality claims

- If the feature claims better convergence: variant `eval/loss` should
  be lower at the same step count, or reach the same loss in fewer
  steps. The effect should be visible, not buried in noise.
- Check loss curves for pathologies: sudden spikes, plateau-then-
  diverge, or suspiciously smooth curves (might indicate a logging
  bug).

#### Red flags

- **Suspiciously good results**: if the variant shows 10x improvement
  on a metric, be skeptical. Check for bugs like accidentally
  disabling noise, training on eval data, or double-counting steps.
- **Identical curves**: if baseline and variant are indistinguishable,
  the feature may not be active. Check that the variant config
  actually enables the feature.
- **Epsilon = 0 or ∞**: accounting is broken.

### Phase 6: Iterate or report

If results are **ambiguous** (small effect, high variance, unexpected
secondary effects), design a follow-up experiment:
- Longer runs to separate signal from noise
- Different hyperparameters to stress-test the feature
- Ablation runs isolating individual components

There is no cap on rounds. Keep iterating until the evidence is clear.

When the evidence is clear, produce a **final report**:

```
## Feature Validation Report: <branch-name>

### Verdict: PASS / FAIL / INCONCLUSIVE

### What was tested
<1-2 sentences describing the change>

### Experiment summary
| Run | W&B link | Steps | Final eval/loss | epsilon | step_time_ms |
|-----|----------|-------|-----------------|---------|--------------|
| ... | ...      | ...   | ...             | ...     | ...          |

### Key findings
<Bullet points with specific numbers>

### DP invariants
- Epsilon: <matched/diverged>
- Grad norms: <within bound/violated>
- Loss stability: <stable/NaN observed at step X>

### Conclusion
<Why this passes or fails, referencing the data>
```

## What this agent does NOT do

- Does not merge or approve PRs
- Does not modify code (it validates, not fixes)
- Does not run local tests (that's `pytest`'s job)
- Does not use fixed pass/fail thresholds — every judgment is
  contextual to what the feature claims

## Prerequisites

- The `jbr-fed-researcher` MCP extension must be installed (provides
  Cadence submission and W&B query skills)
- `HF_TOKEN` and `WANDB_API_KEY` must be configured in Cadence secrets
  (they are by default in the team's `.cadence/configs/` presets)
- The branch must be pushed (Cadence syncs the working directory)
