"""Profiling entry points agree on what an unselected context raises."""

from __future__ import annotations

import pytest

from opaque.api.engine.backend import BackendNotSelectedError, clear_backend
from opaque.profiling import get_memory_stats, reset_peak_memory, step_perf


@pytest.fixture
def _unselected():
    """Drop the root fixture's Torch selection for this test.

    The root `conftest` activates Torch for package suites, so an
    unselected-context assertion has to opt out explicitly or it silently
    tests the selected path instead.
    """
    clear_backend()
    yield
    clear_backend()


@pytest.mark.usefixtures("_unselected")
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: get_memory_stats("cuda"), id="get_memory_stats"),
        pytest.param(lambda: reset_peak_memory("cuda"), id="reset_peak_memory"),
        pytest.param(
            lambda: step_perf("cuda", batch_size=1).__enter__(), id="step_perf"
        ),
    ],
)
def test_unselected_context_raises_the_documented_error(call) -> None:
    """A device string cannot identify a provider, so every entry point raises.

    `"cuda"` and `"cpu"` are not Torch-specific, so profiling must not infer a
    backend from them. It must also not disagree with itself about how it
    declines: `step_perf` probes capabilities before dispatching, and a
    `supports()` probe has nothing to answer with no backend active, so
    without an explicit check it reported a primitive-registration failure
    where its siblings reported `BackendNotSelectedError`.
    """
    with pytest.raises(BackendNotSelectedError):
        call()
