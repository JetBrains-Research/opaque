"""Backend-neutral sampling contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["Sampler"]

T_co = TypeVar("T_co", covariant=True)


class Sampler(Generic[T_co], ABC):
    """Generic base class for objects that yield sampled elements.

    Subclasses must define iteration. Sized samplers may additionally provide
    ``__len__``; the base deliberately leaves length optional.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        """Yield sampled elements."""
        raise NotImplementedError
