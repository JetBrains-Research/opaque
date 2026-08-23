"""Every xse.py env knob must be forwarded into the trainer pod.

WHY THIS TEST EXISTS. deploy/zenml/settings.py forwards campaign-control env vars
via an explicit allowlist. A knob that xse.py reads but that is absent from the
allowlist is SILENTLY IGNORED in the pod: the run completes, logs nothing unusual,
and returns a result bit-identical to the default. On 2026-08-23 that voided three
ablations at once -- XSE_ADAM_STATE, XSE_ADAM_PRECOND and XSE_KEEP_SOURCE -- each
of which was in fact comparing the default configuration against itself, and one of
which had already been reported as a meaningful null result.

The failure is invisible in W&B because these knobs are not argparse flags and so
never reach the run config. This test is the only thing standing between a new knob
and a wasted GPU run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
XSE = REPO / "vendor/lora-privacy/src/lora_privacy/peft_lora_xs/xse.py"
SETTINGS = REPO / "deploy/zenml/settings.py"


def _knobs_read_by_xse() -> set[str]:
    return set(re.findall(r'os\.environ\.get\(\s*"([A-Z_][A-Z0-9_]*)"', XSE.read_text()))


def _knobs_forwarded() -> set[str]:
    text = SETTINGS.read_text()
    start = text.index("_PASSTHROUGH_ENV = (")
    return set(re.findall(r'"([A-Z_][A-Z0-9_]*)"', text[start : text.index(")", start)]))


@pytest.mark.skipif(not XSE.exists(), reason="vendor submodule not checked out")
def test_every_xse_env_knob_is_forwarded_to_the_pod():
    read, forwarded = _knobs_read_by_xse(), _knobs_forwarded()
    missing = sorted(read - forwarded)
    assert not missing, (
        f"xse.py reads {missing} but deploy/zenml/settings.py does not forward them. "
        "A run with one of these set would silently use the DEFAULT and produce a "
        "result identical to the unablated arm. Add them to _PASSTHROUGH_ENV."
    )


@pytest.mark.skipif(not XSE.exists(), reason="vendor submodule not checked out")
def test_allowlist_has_no_dead_entries():
    """A forwarded name xse.py no longer reads is stale and misleading."""
    read, forwarded = _knobs_read_by_xse(), _knobs_forwarded()
    dead = sorted(n for n in forwarded - read if n.startswith("XSE_"))
    assert not dead, f"settings.py forwards {dead}, which xse.py no longer reads"
