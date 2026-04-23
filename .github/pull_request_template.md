<!--
Title (above): Conventional Commits form `<type>(scope): <imperative subject>`.
Examples: `feat(dpsgd): add AdamW-BC`, `fix(accounting): calibrate BLT for beta=0`,
`docs: clarify HF auth env vars`. The PR-gate workflow rejects titles that
don't parse.

Body (below): short, focused prose. On squash merge the body becomes the
commit body that git-cliff reads. Keep it useful for future `git log`
readers.

The `<!-- ai:begin --> ... <!-- ai:end -->` fence below is refreshed on
every push by `pr-describe.yml` — its content is a first-draft
Summary + Test plan. Edit it, replace it, or delete the fence
entirely; the bot won't touch anything outside the markers.
-->

<!-- ai:begin -->
_An AI-drafted summary and test plan will appear here on the first push._
<!-- ai:end -->
