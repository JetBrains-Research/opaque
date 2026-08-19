"""Provider-free serialization tests for MF strategy recipes."""

import numpy as np
import pytest

from opaque.api.dpftrl.noise._strategy_codec import (
    _to_wire,
    deserialize_strategy,
    serialize_strategy,
)
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
)
from opaque.scheduling import linear_schedule


@pytest.mark.parametrize(
    "strategy",
    [
        identity_strategy(),
        band_mf_strategy(bands=3, momentum=0.8),
        blt_strategy(max_buffers=3, momentum=0.9),
        bsr_strategy(bandwidth=3, alpha=1.0, beta=0.1),
        bisr_strategy(bandwidth=3, inv_coefficients=(1.0, -0.2, -0.1)),
        lambda_cgd_strategy(lambda_=0.4),
    ],
)
def test_strategy_recipe_round_trip(strategy):
    restored = deserialize_strategy(serialize_strategy(strategy))
    assert restored == strategy


def test_schedule_recipe_round_trip():
    schedule = linear_schedule(init_value=1.0, end_value=0.1, transition_steps=20)
    strategy = band_mf_strategy(bands=3, lr_schedule=schedule)
    restored = deserialize_strategy(serialize_strategy(strategy))

    assert [restored.lr_schedule(step) for step in (0, 10, 20)] == pytest.approx(
        [schedule(step) for step in (0, 10, 20)]
    )


def test_numpy_arrays_encode_as_plain_wire_lists():
    assert _to_wire(np.asarray([1.0, 2.0], dtype=np.float64)) == [1.0, 2.0]
