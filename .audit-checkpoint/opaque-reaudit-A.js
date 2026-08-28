export const meta = {
  name: 'opaque-reaudit-finders',
  description: 'Phase A: verify remediation of the July audit and sweep the changed library — finder agents only',
  phases: [
    { title: 'FixVerify', detail: '12 subsystem agents verify all 95 closed audit issues and review the full delta' },
    { title: 'NewSweep', detail: '10 agents audit surfaces added or rewritten since the audit' },
  ],
}

const SP = '/tmp/claude-0/-home-user-opaque/2c6b9515-9455-539f-b005-ebecde2ecc66/scratchpad'
const BASE = '79c916e3'
const HEAD = '4b13d82'

const SHARED = `You are auditing Opaque, a differential-privacy training library for PyTorch, at /home/user/opaque (checked out at origin/main, commit ${HEAD}).

CONTEXT. A full audit completed on 2026-07-29 found 209 issues (IDs OPQ-001..). The condensed master list with per-OPQ evidence and verdicts: ${SP}/AUDIT-master-list.md (recovered from the repo's own git history; Read it and filter for your area; the remediation plan is ${SP}/AUDIT-remediation-plan.md). The team then spent a month remediating: 202 commits landed between ${BASE} (pre-remediation tip) and ${HEAD}, including releases v0.4.0 through v0.9.0. The audit backlog became GitHub issues: ${SP}/issues-closed-audit.tsv lists the 95 CLOSED (claimed-fixed) issues; ${SP}/issues-open.tsv lists the 48 still-OPEN (tracked outstanding) issues. ${SP}/delta-commits.txt lists all 202 delta commits. GitHub API access is unavailable in this session — work from git history and the TSVs.

GIT. Inspect the remediation itself: 'git log --oneline ${BASE}..${HEAD} -- <path>' scopes delta commits to a path; 'git show <sha>' shows one change; 'git diff ${BASE}..${HEAD} -- <path>' shows the net delta. The working tree IS ${HEAD}.

RULES.
- Evidence or it does not exist: every claim must cite file:line from the CURRENT tree (plus commit SHAs where relevant). Quote the actual code. Do not trust commit messages, docstrings, comments, or docs as evidence that something works — they state intent, not behavior.
- Do NOT report anything that merely duplicates an OPEN issue — that work is tracked. If you find an open issue's problem is materially WORSE than its title suggests, report that delta as a finding and name the issue number in the claim.
- Fixes are the least-reviewed code in the repo. A fix can be incomplete (misses call sites), wrong (fixes the symptom, not the invariant), or regressive (breaks something that worked). Hunt for all three. Also watch for the audited defect pattern surviving in SIBLING code paths the fix did not touch.
- Severity: critical = silent privacy loss or silent training corruption with defaults; high = wrong epsilon/results in realistic configs, or fail-open behavior; medium = wrong under specific configs, misleading docs on safety-relevant behavior; low = everything else worth fixing.
- affects_epsilon=true iff the issue can change the real or reported privacy guarantee.
- Read-only on the repo: do not edit repo files. Running existing tests or small Python/Rust snippets IS encouraged ('uv run python' or python3 with PYTHONPATH into packages/*/src). Do NOT run 'uv sync' variants that download GPU wheels; if an import needs an uninstalled heavy dependency, note it and move on.
- Confidence: high only if you traced the full path from entry point to defect and checked for guards upstream.

DURABILITY. This session has crashed twice; your returned result can be lost. As your FINAL action before producing structured output, Write your complete result as JSON to ${SP}/results/<your-label>.json (your label is given in your role line). Then return the same content as structured output.`

const FINDING_PROPS = {
  title: { type: 'string' },
  file: { type: 'string' },
  line: { type: 'string' },
  category: { type: 'string' },
  severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
  claim: { type: 'string' },
  failure_scenario: { type: 'string' },
  evidence: { type: 'array', items: { type: 'string' } },
  affects_epsilon: { type: 'boolean' },
  fix_sketch: { type: 'string' },
  confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
}
const FINDING_REQ = ['title', 'file', 'line', 'category', 'severity', 'claim', 'failure_scenario', 'evidence', 'affects_epsilon', 'fix_sketch', 'confidence']

const FIXV_SCHEMA = {
  type: 'object',
  properties: {
    statuses: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          issue: { type: 'integer' },
          title: { type: 'string' },
          status: { type: 'string', enum: ['fixed-verified', 'fixed-with-caveats', 'partial', 'not-fixed', 'regressed', 'cannot-verify'] },
          fix_commits: { type: 'array', items: { type: 'string' } },
          note: { type: 'string' },
        },
        required: ['issue', 'title', 'status', 'note'],
      },
    },
    findings: { type: 'array', items: { type: 'object', properties: FINDING_PROPS, required: FINDING_REQ } },
    delta_reviewed: { type: 'string' },
  },
  required: ['statuses', 'findings', 'delta_reviewed'],
}

const SWEEP_SCHEMA = {
  type: 'object',
  properties: {
    findings: { type: 'array', items: { type: 'object', properties: FINDING_PROPS, required: FINDING_REQ } },
    clean_areas: { type: 'array', items: { type: 'string' } },
  },
  required: ['findings', 'clean_areas'],
}

const SUBSYSTEMS = [
  { key: 'accounting-py', labels: 'accounting', paths: 'packages/opaque-accounting/src (Python only) and its tests', focus: 'PLD composition, CachedProcess/repeated_pld (the headline audit bug), budget direction (#333 was the inverted budget_exceeded), serialization/equality (#334, #342), calibration (#333), mixture constants (#339), FFT fallback (#340), composition-count validation (#341), PLD cache keying (#332), neg-inf mass (#338). Re-derive the CachedProcess fix from scratch: does cached(per_step(p)) * K now equal per_step(p) * K bit-for-bit? Run it. Known from a partial prior pass (verify anew, statuses were: 331/333/336/337/339/340/341 fixed-verified; 332/334 fixed-with-caveats — 332 residual: strategy dataclasses declare lr_schedule compare=False so eq/hash stay schedule-blind and the composition MERGE optimizer can collapse distinct-schedule processes; 334 residual: wire format is a nested dict so json.dump at the trainer checkpoint path still hits RecursionError at ~1000-deep spines; 335/338 not-fixed/withdrawn — 335 has NO scheduled workflow at all; 331 pattern survives in the b-min-sep MC driver).' },
  { key: 'accounting-rs', labels: 'accounting', paths: 'packages/opaque-accounting Rust crate (src/*.rs, monte_carlo, adaclip, discretization)', focus: 'Rounding directions after the remediation (the original truncation bug was unsafe-direction rounding sold as conservative), Monte Carlo bounding (#337, #331, PR #748 — prior pass confirmed a per-rank KL/Chernoff lower confidence band with union over ranks/adjacency, residual pushed to infinity_mass, fail-closed beta; re-verify and go deeper), adaclip sensitivity (#345 still open — check the Rust side has not silently changed), Rust lints (#422), FFI boundary. Verify any new conservative-bound math against first principles: direction of every inequality. The b-min-sep MC driver reportedly still has the thread-count nondeterminism pattern #331 fixed for balls-in-bins — confirm and characterize.' },
  { key: 'dpsgd', labels: 'dpsgd', paths: 'packages/opaque-dpsgd', focus: 'Poisson/parallel-Poisson samplers and validation (#336, #354), the critical sync(aux) data-dependent-collectives fix, adaptive clipping normalization vs accounting (#345-adjacent), bounded Gaussian (#344 still open — verify runtime behavior unchanged or gated), random-allocation PLD transform (PR #307, new feature).' },
  { key: 'dpftrl', labels: 'dpftrl', paths: 'packages/opaque-dpftrl', focus: 'lambda-CGD zero-noise-at-final-step fix, BISR full operator (#360), balls-in-bins empty slots (#358), sampler-across-epochs (#357), MF noise horizon guard, column-keyed draws (#361), BLT min-sep (#595 open — verify scope), lr_schedule in strategy identity (#332/#362), streaming noise continuity across resume (#426).' },
  { key: 'engine-core', labels: 'engine', paths: 'packages/opaque-engine (non-distributed)', focus: 'all_finite through wrappers (#352), loss-scaler skipped-step accounting (#350), microbatch/bf16 clipping numerics (#343), serialization of tensor subclasses (the nn.Parameter drop bug), checkpoint shape validation (#349), PerGroup immutability (#347), adaptive clipping memory (#351).' },
  { key: 'engine-dist', labels: 'distributed', paths: 'distributed code across opaque-engine, opaque-dpsgd, opaque-transformers', focus: 'the critical empty-Poisson-batch collective desync fix — is the collective count now schema-derived (#368) or still data-dependent anywhere; paired second-moment reduction (#367), scalar reduction exactness (#365), collective retry (#363), uneven eval gather (#369), two-rank CI lane (#366, #328) — does the lane actually run and would it catch the original desync?' },
  { key: 'optimizers', labels: 'optimizers', paths: 'packages/opaque-optimizers', focus: 'Adafactor eps floors scale-aware (#411) — is the new formulation actually scale-invariant or just a different constant; noise-variance EMA restore (#410), update_rms_clip semantics (#409), schedule-free accessor (#408), memory claims (#406), test vacuousness (#405), HF fused-optimizer translation (#389).' },
  { key: 'alignment', labels: 'alignment', paths: 'packages/opaque-alignment', focus: 'Gemma chat-template span fix (#387) — test against actual Gemma/Llama/ChatML templates, not just the fixed case; DPO cache fingerprint (#379) and TR-DPO temp cache (#391), BCO/MPO/f-div/LD-DPO contracts (#394), DiscoPOP clamp (#395), SquareChiPO disposition (#393), fused/eager dtype (#390), PEFT ref-model adapters (#381), ref-logprob sharding (#364).' },
  { key: 'transformers', labels: 'transformers', paths: 'packages/opaque-transformers', focus: 'DPTrainer: stop-at-epsilon every accounted step (#392), RNG stream separation between adaclip quantile noise and gradient noise (the byte-identical streams bug), inert args (#388), DFT eval objective (#384), fractional epochs (#382), best-model boundaries (#386), eval cadence (#385), calibrated complexity limits (PR #747), metadata/extras (#430). Also the un-noised logging statistics — #383 is still open, but check nothing NEW leaks.' },
  { key: 'patches', labels: 'patches', paths: 'packages/opaque-patches', focus: 'sliding-window attention fix (#397) — verify semantics per model family, not just that a mask exists; Gemma2 softcapping (#398), CE-backward logits overwrite (#402), LoRA fused validation (#399, #401), router dropout (#400), checkpoint-override scoping (#396), parity harness (#404, #330 open — what landed), version-aware patches for torch 2.12 (PR #268): version gates correct? upstream parity for each gated branch?' },
  { key: 'auditing', labels: 'auditing', paths: 'packages/opaque-auditing', focus: 'NaN membership scores fix, infinite-score ROC denominators (#378), GDP grid rejection (#373), score ordering (#371), sorted batch_argnums (#372), RNG domain separation (#374), canary construction (#377) and caller-selected canary pools (PR #751, new), perfect-separation tests (#370). The GdpMethod._mu_at hang: verify termination on the exact pathological inputs from the audit.' },
  { key: 'packaging-ci', labels: 'ci OR packaging OR docs OR api', paths: 'pyproject.toml files, uv.lock, .github/workflows, release artifacts, third_party_provenance.toml, umbrella package', focus: 'reproducible accounting artifacts (#421), pinned umbrella deps (#420), licenses/attribution (#423), lockfile dependabot coverage (#425, PR #752), py3.12 lane (#428), Triton validation blocking (#424), TRL parity CI (#431), executed-docs CI (#416), citation CI (#418), nightly privacy-regression vectors (#335 — prior pass found NO scheduled workflow exists; confirm and check what analytic vector coverage runs in ordinary lanes). Releases v0.4.0-v0.9.0: verify tags build from main history.' },
]

const SWEEPS = [
  { key: 'horizon-prefix', focus: 'The horizon-prefix accounting machinery is NEW since the audit (grep for prefix/horizon/_pld_at_horizon/partial across opaque-accounting and opaque-dpftrl; see open issues #683-#700 for known softness — do NOT re-report those). Audit what LANDED: prefix PLD construction for each strategy family, monotonicity of epsilon in the prefix length (a non-monotone prefix bound is exploitable via early stopping), the interaction with cached/Repeated wrappers (#685 open — check adjacent code), and whether mid-run epsilon_at bounds the RELEASED transcript or the planned horizon. Look for conservative-in-name-only bounds: any place a prefix is priced with full-horizon correlation structure but realized with a truncated one.' },
  { key: 'mc-bounds', focus: 'PR #748 "bound Monte Carlo privacy guarantees" and its Rust/Python implementation (follow-up to #337/#331; #666 and #697 remain open — skip their scope). Verify the statistical construction from first principles: what concentration inequality converts samples to a bound, is it applied to the right functional, one-sided in the safe direction, does the confidence parameter reach the user-facing API and docs? Check reproducibility across thread counts and seeds. Write and run a small numerical check if feasible.' },
  { key: 'random-allocation', focus: 'PR #307 added a random-allocation PLD transform for DP-SGD (new feature, never audited). Audit it fully: math vs the cited paper (WebFetch the paper if referenced), edge cases (n=1, k=n, k=0), interaction with amplification and composition, whether the runtime sampler it models is the one that exists, serialization round-trip.' },
  { key: 'canary-pools', focus: 'PR #751 "support caller-selected canary pools" in opaque-auditing (new, never audited). Audit: does pool selection bias the audit statistic (selection after seeing scores = invalid), RNG handling, whether the epsilon-lower-bound machinery conditions correctly on the pool, API misuse paths (empty pools, duplicate canaries, pool overlapping training data), docs claims.' },
  { key: 'mps-backend', focus: 'PR #267 stabilized MPS (Apple Silicon) support: device caps, profiling, bf16, compile. Plus any backend-split refactor that merged (check git log for backend-split). Audit: silent fallbacks (MPS lacking an op -> CPU fallback or wrong dtype without warning), RNG determinism of noise generation on MPS vs CPU/CUDA (privacy noise MUST be full-precision and correctly seeded per device), float64 unavailability on MPS silently degrading accounting or clipping precision, bf16 clipping numerics, torch.compile graph breaks changing numerics.' },
  { key: 'trainer-calib', focus: 'PR #747 "enforce calibrated complexity limits" in opaque-transformers, plus fail-closed calibration and estimate-vs-bound distinction end-to-end in the trainer: when calibration cannot meet the target, what happens now — hard error, or a fallback that trains anyway? Trace every path from privacy_target_epsilon to the noise actually applied at step 1, including resume-from-checkpoint with the fixed CachedProcess semantics. The July audit found calibration and runtime accounting disagreed; verify they now agree bit-for-bit, and run a check.' },
  { key: 'test-integrity', focus: 'PR #275 parallelized the test suite; #759 just consolidated shared test infrastructure; #753 renamed comparisons. Audit the TESTS: shared-state hazards under parallel execution (tmp paths, global RNG seeding, env vars, port collisions), whether any test was weakened/skipped/xfailed during the month (git log -p on test files: search for added skip/xfail/relaxed tolerances), and for the ORIGINAL headline audit bugs (CachedProcess repeated_pld, lambda-CGD final-step zero noise, budget_exceeded inversion, Adafactor floors) determine whether the CURRENT suite would fail on the OLD buggy code. Name each regression test that would/would not catch it.' },
  { key: 'docs-drift', focus: 'Docs were heavily rewritten (limitations.md is new). Sweep docs/ + README + package READMEs + public-API docstrings for claims now wrong AGAINST the post-remediation code: defaults, guarantees, "we account for X" statements, worked examples with numbers, the sampler-vs-accountant story (#356 still open — is the docs story honest about it?), limitations.md completeness against the high-severity open issues (#344, #345, #359, #383, #595).' },
  { key: 'citations', focus: 'Citation integrity on the CURRENT tree (#418 closed claims scholarly references were repaired). Extract every paper citation from docs and docstrings (grep for arXiv/DOI/author-year), verify each against the actual paper (WebFetch arxiv.org/abs pages and PDFs): does the cited theorem state what the code claims, are constants/conditions preserved, any theorem applied outside its hypotheses (the July audit found truncation sold as conservative citing a paper that required the opposite). Prioritize accounting/dpftrl/auditing citations — they carry the privacy proofs.' },
  { key: 'api-misuse', focus: 'Fresh API-misuse sweep of the post-remediation public surface: enumerate the exported API of each package (__init__/facades), then hunt for foot-guns a competent user hits: kwargs silently ignored, mutable defaults, functions that accept-and-ignore invalid combos, error messages pointing at the wrong knob, renamed-but-aliased APIs where old and new silently diverge, deprecation shims that changed semantics. The remediation renamed/moved a lot (202 commits) — migration seams are where silent breakage lives. Also check the umbrella package re-exports match subpackage reality.' },
]

function fixPrompt(s) {
  return `${SHARED}

ROLE (label: fixverify-${s.key}): Fix verification for the ${s.key} area: ${s.paths}.

TASK.
1. Read ${SP}/issues-closed-audit.tsv and take every row whose labels match: ${s.labels}. For EACH such closed issue, verify the claimed fix against the current tree. Find the fix commits ('git log --oneline ${BASE}..${HEAD} --grep="#<issue>"' and by content). Establish status: fixed-verified (traced and confirmed at every call site), fixed-with-caveats (works but with a residual defect — which you MUST also report as a finding), partial, not-fixed, regressed (also a finding), or cannot-verify (say exactly what blocked you). Cross-reference OPQ entries in ${SP}/AUDIT-master-list.md for the precise original defect definitions where present — verify against THOSE, not the issue title.
2. Review the ENTIRE delta in your area, not only fix commits: 'git log --oneline ${BASE}..${HEAD} -- <your paths>'. Report defects in features/refactors as findings. Summarize coverage in delta_reviewed.
3. Specific attention: ${s.focus}

Write your JSON to ${SP}/results/fixverify-${s.key}.json, then return it as structured output. An empty findings list from a real verification is fine; a status you did not verify is not.`
}

function sweepPrompt(s) {
  return `${SHARED}

ROLE (label: sweep-${s.key}): New-surface audit: ${s.key}.

TASK. ${s.focus}

This surface is new or heavily rewritten since the July audit and has never been adversarially reviewed. Go deep rather than wide: trace complete paths, run code where feasible, derive the math yourself where the code claims a bound. Report findings with full evidence chains; list what you checked and found sound in clean_areas (specifically — 'checked X property by method Y').

Write your JSON to ${SP}/results/sweep-${s.key}.json, then return it as structured output.`
}

log(`FixVerify: ${SUBSYSTEMS.length} agents; NewSweep: ${SWEEPS.length} agents (phase A only; verification runs as a separate workflow)`)

const all = await parallel([
  ...SUBSYSTEMS.map(s => () => agent(fixPrompt(s), { label: `fixverify:${s.key}`, phase: 'FixVerify', schema: FIXV_SCHEMA })),
  ...SWEEPS.map(s => () => agent(sweepPrompt(s), { label: `sweep:${s.key}`, phase: 'NewSweep', schema: SWEEP_SCHEMA })),
])

const fixResults = all.slice(0, SUBSYSTEMS.length).map((r, i) => r && { area: SUBSYSTEMS[i].key, ...r }).filter(Boolean)
const sweepResults = all.slice(SUBSYSTEMS.length).map((r, i) => r && { area: SWEEPS[i].key, ...r }).filter(Boolean)

const rawFindings = []
for (const r of fixResults) for (const f of r.findings || []) rawFindings.push({ ...f, source: `fixverify:${r.area}` })
for (const r of sweepResults) for (const f of r.findings || []) rawFindings.push({ ...f, source: `sweep:${r.area}` })

const statusRows = []
for (const r of fixResults) for (const s of r.statuses || []) statusRows.push({ area: r.area, ...s })

return {
  stats: {
    fixAgentsDone: fixResults.length,
    sweepAgentsDone: sweepResults.length,
    statuses: statusRows.length,
    rawFindings: rawFindings.length,
    statusBreakdown: statusRows.reduce((m, s) => { m[s.status] = (m[s.status] || 0) + 1; return m }, {}),
  },
  statuses: statusRows,
  findings: rawFindings,
  deltaNotes: fixResults.map(r => ({ area: r.area, delta_reviewed: r.delta_reviewed })),
  cleanAreas: sweepResults.map(r => ({ area: r.area, clean_areas: r.clean_areas })),
}
