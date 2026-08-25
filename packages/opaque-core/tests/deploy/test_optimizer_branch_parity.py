"""The frozen and rotating optimizer branches must agree on every shared knob.

WHY THIS EXISTS. On 2026-08-25 a workflow found that examples/train_causal_lm.py
built the frozen arm as `sgd(lr=lr_for_opt, weight_decay=...)`, omitting momentum,
which opaque.optimizers.sgd defaults to 0.0 -- while the rotating arm passed
`momentum=args.sgd_momentum` (0.9). So every frozen-vs-rotating comparison on the
SGD path was plain SGD against heavy-ball, worth about 2.7e-3 of an 8.9e-3 headline.

It was invisible in W&B: the run config records the ARGPARSE value, so both arms
logged sgd_momentum=0.9 regardless of what the optimizer received. Checking configs
could not detect it. Only reading the two branches side by side could -- which is
what this test automates.

Fourth defect of this shape in the campaign (env-var allowlist, submodule pointer,
run.py argv precedence, and now this), so it is worth a standing check.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
TRAINER = REPO / "examples/train_causal_lm.py"

# knobs the two branches must both honour when the optimizer supports them
SHARED_KNOBS = ("momentum", "weight_decay")


def _branch_source(name: str) -> str:
    """Text of the call that constructs `name`(...), parens-balanced."""
    src = TRAINER.read_text()
    i = src.index(f"{name}(")
    depth, j = 0, i + len(name)
    while j < len(src):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                return src[i : j + 1]
        j += 1
    raise AssertionError(f"unbalanced parens after {name}(")


@pytest.mark.skipif(not TRAINER.exists(), reason="trainer not present")
def test_frozen_sgd_passes_momentum():
    """The plain-sgd branch must pass momentum, or it silently differs from xse_sgd."""
    call = _branch_source("base_opt = sgd")
    assert "momentum" in call, (
        "the frozen sgd branch does not pass momentum; opaque.optimizers.sgd "
        "defaults it to 0.0, so the frozen arm would train as plain SGD while the "
        f"rotating arm uses heavy-ball. Call was:\n{call}"
    )
    assert "args.sgd_momentum" in call, (
        f"momentum is passed but not from args.sgd_momentum:\n{call}"
    )


@pytest.mark.skipif(not TRAINER.exists(), reason="trainer not present")
def test_both_adamw_branches_pass_betas():
    """Both AdamW branches must take betas from the same args, or beta1 diverges."""
    src = TRAINER.read_text()
    n = len(re.findall(r"betas=\(args\.sgd_momentum,\s*args\.adam_beta2\)", src))
    assert n >= 2, (
        f"expected both AdamW branches to pass betas=(args.sgd_momentum, "
        f"args.adam_beta2); found {n} such call(s)"
    )
