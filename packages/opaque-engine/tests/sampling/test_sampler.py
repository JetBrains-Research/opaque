from collections.abc import Iterator

import pytest

from opaque.api.engine.sampling import Sampler as ImplementationSampler
from opaque.sampling import Sampler


class FiniteSampler(Sampler[int]):
    def __iter__(self) -> Iterator[int]:
        yield from (1, 2, 3)

    def __len__(self) -> int:
        return 3


class UnsizedSampler(Sampler[str]):
    def __iter__(self) -> Iterator[str]:
        yield "sample"


def test_facade_exports_engine_sampler() -> None:
    assert Sampler is ImplementationSampler


def test_sampler_supports_generic_iteration_and_optional_length() -> None:
    finite = FiniteSampler()
    unsized = UnsizedSampler()

    assert list(finite) == [1, 2, 3]
    assert len(finite) == 3
    assert list(unsized) == ["sample"]
    with pytest.raises(TypeError, match="has no len"):
        len(unsized)


def test_sampler_requires_iteration_implementation() -> None:
    class IncompleteSampler(Sampler[int]):
        pass

    # Python 3.12 quotes the method name in this message and 3.11 does not,
    # so match around the quoting rather than pinning one interpreter.
    with pytest.raises(TypeError, match=r"abstract method .?__iter__"):
        IncompleteSampler()
