# Mellum2 under Opaque DP: research artefacts

Companion directory of `../mellum-dp-research.md` (the report). Layout:

- `design-spec.md`: the full design specification (v2, post-refutation). Read its sections 9 and 10 before implementing. Errata are listed at its top.
- `reports/`: the agent reports of each research phase (understanding, design panel with judges and adversarial refutation, validation).
- `scripts/phase{1,2,3}/`: every experiment script with its captured output. All ran on CPU in the repository's `uv` environment (`uv run python <script>`); paths inside the scripts point at the original scratch directory and may need adjusting.

Nothing here is part of the library. The scripts use internal APIs and tiny random-init models; they are evidence for the report, not tests.
