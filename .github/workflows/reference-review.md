---
description: "Review changed references in pull requests using GitHub Agentic Workflows"
on:
  pull_request:
    branches: [main]
    types: [opened, synchronize, reopened, ready_for_review]
permissions:
  contents: read
  pull-requests: read
  copilot-requests: write
  actions: read
engine: copilot
tools:
  bash: [python3, git]
  github:
    mode: gh-proxy
    toolsets: [actions]
jobs:
  collect-reference-context:
    runs-on: ubuntu-latest
    outputs:
      changed_files: ${{ steps.collect.outputs.changed_files }}
      reference_entries: ${{ steps.collect.outputs.reference_entries }}
    steps:
      - uses: actions/checkout@v5
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v6
        with:
          python-version: "3.11"
      - name: Collect reference review context
        id: collect
        run: |
          python3 .github/scripts/collect_reference_review_context.py \
            --base "${{ github.event.pull_request.base.sha }}" \
            --output /tmp/reference_context.json
          python3 - <<'PY'
          import json, os
          from pathlib import Path
          data = json.loads(Path("/tmp/reference_context.json").read_text())
          with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
              fh.write(f"changed_files={json.dumps(data['files'])}\n")
              fh.write(f"reference_entries={json.dumps(data['entries'])}\n")
          PY
safe-outputs:
  create-pull-request-review-comment:
    max: 20
    target: "triggering"
  submit-pull-request-review:
    allowed-events: [COMMENT]
    target: "triggering"
---

Review the changed pull request diff for free-form paper references.

Use the precomputed review context from the `collect-reference-context` job:

- changed files: `${{ needs.collect-reference-context.outputs.changed_files }}`
- reference-like lines: `${{ needs.collect-reference-context.outputs.reference_entries }}`

You are not required to rely on a local citation registry. Search the repository
for nearby mentions of the same paper, title, arXiv id, DOI, or author/year
pattern to infer the dominant canonical wording already used in Opaque.

Your goal is **high-precision review**, not exhaustive citation cleanup.

For each changed reference that looks questionable:

1. Compare it against nearby repository usage first.
2. If needed, infer the likely normalized form from the cited arXiv id, DOI, or
   repeated repo-local wording.
3. Flag likely mismatches in title, author attribution, year, or identifier.
4. Prefer inline review comments on the changed diff lines instead of a single
   summary comment.

Comment only when confidence is high. In particular:

- Treat arXiv-id or DOI mismatches as the strongest signal.
- Prefer existing repository wording over inventing a new normalized citation.
- Do not comment on generic mentions like `Smith et al. 2024` unless the line
  carries enough context to identify a specific paper confidently.
- If multiple plausible papers match, skip the line.
- If the line is acceptable but stylistically different, skip it.
- Avoid repository-wide cleanup advice; comment only on the changed line.

When you leave a comment:

- Quote the conflicting text briefly.
- State the likely issue in one sentence.
- Suggest a concrete replacement if you can do so confidently.

Ignore lines that already look acceptable or are too ambiguous to judge
confidently.
