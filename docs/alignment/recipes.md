# Alignment recipes

This page is a placeholder for end-to-end DP-alignment recipes built on the
`opaque-alignment` primitives. The functional examples in `examples/` are the
current starting points:

- `examples/train_sft.py` — DP-SGD supervised fine-tuning.
- `examples/train_dpo.py` — DP-SGD DPO with precomputed reference logps.
- `examples/train_kto.py` — DP-SGD KTO (Tier-2 detached-KL caller pattern).

Each script swaps the DP mechanism at a single call site: import a noise
mechanism from `opaque.dpsgd.noise` (Gaussian) or `opaque.dpftrl.noise`
(matrix-factorized / correlated noise) and an optimizer from
`opaque.optimizers`. The loss closure is identical across mechanisms.

## Planned recipes

Future additions (see the package plan, §14 roadmap):

- **Decoupled DP-RLHF** — DP reward-model training feeding a (non-DP) PPO actor.
- **SquareχPO defaults** — the first optimal-rate DP-DPO recipe
  (`dpo_squarechipo` + DP-AdamW).
- **Recipe DSL** — `@register_recipe("sft+dpo")`-style registration for paper
  recipes.
