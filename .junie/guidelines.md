# Opaque development guidelines

Before planning, implementing, or reviewing a change:

1. Read `AGENTS.md` for repository layout, commands, and workflow.
2. Read `docs/development/architecture-contracts.md` and apply every normative
   active contract whose scope intersects the change.
3. Treat the advisory rules in that document as design feedback, not automatic
   blockers.
4. Do not report missing infrastructure for contracts marked planned.

For implementation work, preserve the architecture contracts while making the
smallest complete change. Use the owning package's public façade in user-facing
code and its `opaque.api.*` implementation tree for implementation. Add or move
tests according to ARC-006 and run the smallest relevant existing validation.

For code review:

- Review the merge-base diff, not an exhaustive inventory of the repository.
- Post only actionable, high-confidence findings on changed lines.
- Cite the relevant `ARC-*` or `ADV-*` ID for architecture findings.
- Distinguish normative violations from advisory design suggestions.
- Do not flag a new package, module, import, or export merely because it is new.
- Do not replace semantic reasoning with a hard-coded package/module list.

If a requested change intentionally revises an architecture contract, stop and
propose the policy change explicitly instead of silently working around it.
