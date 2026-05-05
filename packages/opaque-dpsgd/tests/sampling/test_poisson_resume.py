"""Resume / state_dict tests for Poisson and TruncatedPoisson samplers."""

from __future__ import annotations

import json

from opaque.dpsgd.sampling import PoissonSampler, TruncatedPoissonSampler
from opaque.random import RngKey, key


def _drain(it, n):
    out = []
    for _ in range(n):
        out.append(next(it))
    return out


class TestPoissonSamplerResume:
    def test_state_dict_roundtrip_via_json(self):
        s = PoissonSampler(
            data_source=range(100),
            sample_rate=0.1,
            num_iterations=20,
            key=key(7),
        )
        state = s.state_dict()
        encoded = json.dumps(state)
        decoded = json.loads(encoded)
        assert decoded == state

    def test_initial_state(self):
        s = PoissonSampler(
            data_source=range(10),
            sample_rate=0.5,
            num_iterations=5,
            key=key(42),
        )
        st = s.state_dict()
        assert st["iter_count"] == 0
        assert st["sample_rate"] == 0.5
        assert st["num_iterations"] == 5
        assert st["key"]["seed"] == key(42).seed

    def test_resume_produces_same_sequence(self):
        """Save mid-iteration → recreate → load → next batches match the original sequence."""
        sampler_a = PoissonSampler(
            data_source=range(200),
            sample_rate=0.05,
            num_iterations=10,
            key=key(123),
        )
        all_batches_a = list(sampler_a)
        assert len(all_batches_a) == 10

        # Restart, drain 4, snapshot, then create a new sampler and resume.
        sampler_b = PoissonSampler(
            data_source=range(200),
            sample_rate=0.05,
            num_iterations=10,
            key=key(123),
        )
        it_b = iter(sampler_b)
        first_four = _drain(it_b, 4)
        assert first_four == all_batches_a[:4]
        snapshot = sampler_b.state_dict()

        sampler_c = PoissonSampler(
            data_source=range(200),
            sample_rate=0.5,  # placeholder — will be overwritten
            num_iterations=1,
            key=key(0),
        )
        sampler_c.load_state_dict(snapshot)
        rest = list(sampler_c)
        assert rest == all_batches_a[4:]

    def test_load_state_dict_overwrites_fields(self):
        sampler = PoissonSampler(
            data_source=range(50),
            sample_rate=0.5,
            num_iterations=5,
            key=key(0),
        )
        sampler.load_state_dict({
            "key": {"seed": 999, "impl": "opaque_threefry_like"},
            "iter_count": 3,
            "sample_rate": 0.2,
            "num_iterations": 10,
        })
        assert sampler.sample_rate == 0.2
        assert sampler.num_iterations == 10
        assert sampler._iter_count == 3
        assert sampler._key == RngKey(seed=999, impl="opaque_threefry_like")

    def test_iteration_advances_state(self):
        sampler = PoissonSampler(
            data_source=range(20),
            sample_rate=0.5,
            num_iterations=3,
            key=key(11),
        )
        assert sampler.state_dict()["iter_count"] == 0
        list(sampler)  # exhaust
        assert sampler.state_dict()["iter_count"] == 3

    def test_unbounded_sampler_state_dict(self):
        sampler = PoissonSampler(
            data_source=range(10),
            sample_rate=0.5,
            num_iterations=None,
            key=key(1),
        )
        st = sampler.state_dict()
        assert st["num_iterations"] is None
        # Should round-trip
        sampler2 = PoissonSampler(
            data_source=range(10),
            sample_rate=0.1,
            num_iterations=5,
            key=key(0),
        )
        sampler2.load_state_dict(st)
        assert sampler2.num_iterations is None


class TestTruncatedPoissonSamplerResume:
    def test_state_dict_carries_max_batch_size(self):
        s = TruncatedPoissonSampler(
            data_source=range(100),
            sample_rate=0.5,
            max_batch_size=8,
            num_iterations=5,
            key=key(3),
        )
        st = s.state_dict()
        assert st["max_batch_size"] == 8

    def test_resume_produces_same_sequence(self):
        sampler_a = TruncatedPoissonSampler(
            data_source=range(100),
            sample_rate=0.5,
            max_batch_size=10,
            num_iterations=8,
            key=key(99),
        )
        all_batches_a = list(sampler_a)

        sampler_b = TruncatedPoissonSampler(
            data_source=range(100),
            sample_rate=0.5,
            max_batch_size=10,
            num_iterations=8,
            key=key(99),
        )
        it = iter(sampler_b)
        first_three = _drain(it, 3)
        assert first_three == all_batches_a[:3]
        snapshot = sampler_b.state_dict()

        sampler_c = TruncatedPoissonSampler(
            data_source=range(100),
            sample_rate=0.5,
            max_batch_size=10,
            num_iterations=8,
            key=key(99),
        )
        sampler_c.load_state_dict(snapshot)
        rest = list(sampler_c)
        assert rest == all_batches_a[3:]
        # batch sizes never exceed max
        for batch in all_batches_a:
            assert len(batch) <= 10

    def test_truncation_uses_per_iter_generator(self):
        """Two independent samplers with the same key produce identical truncated batches."""
        a = TruncatedPoissonSampler(
            data_source=range(200),
            sample_rate=0.5,
            max_batch_size=5,
            num_iterations=4,
            key=key(7),
        )
        b = TruncatedPoissonSampler(
            data_source=range(200),
            sample_rate=0.5,
            max_batch_size=5,
            num_iterations=4,
            key=key(7),
        )
        assert list(a) == list(b)
