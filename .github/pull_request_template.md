<!--
Title (above): Conventional Commits form `<type>(scope): <imperative subject>`.
Examples: `feat(dpsgd): add AdamW-BC`, `fix(accounting): calibrate BLT for beta=0`,
`docs: clarify HF auth env vars`. The PR-gate workflow rejects titles that
don't parse.

Body (below): short, focused prose. On squash merge the body becomes the
commit body that git-cliff reads. Keep it useful for future `git log`
readers.
-->

## Summary

<!-- Why this change exists, and what it does, in 2–4 sentences. -->

## Test plan

<!-- Commands run, edge cases verified, behavior expected. -->

## Checklist

- [ ] Tests added/updated
- [ ] Docstrings / user guides updated for user-facing changes
- [ ] DP guarantees preserved for DP-related changes
