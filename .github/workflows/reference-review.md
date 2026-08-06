---
description: "Review changed references in pull requests using GitHub Agentic Workflows"
on:
  pull_request:
    branches: [main]
    types: [opened, synchronize, reopened, ready_for_review]
permissions:
  contents: read
  pull-requests: write
tools:
  bash:
    - python3 .github/scripts/collect_reference_review_context.py --base "${{ github.event.pull_request.base.sha }}" --output /tmp/reference_context.json
  github:
    pull-requests: read
    pull-request-review-comments: write
steps:
  - name: Check out repository
    uses: actions/checkout@v5
    with:
      fetch-depth: 0
  - name: Set up Python
    uses: actions/setup-python@v6
    with:
      python-version: "3.11"
artifacts:
  - /tmp/reference_context.json
---

Review the changed pull request diff for free-form paper references.

Use `.github/scripts/collect_reference_review_context.py` output from
`/tmp/reference_context.json` as your review context. It contains:

- changed files under `README.md`, `docs/`, and `packages/`;
- reference-like lines extracted from those files.

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
