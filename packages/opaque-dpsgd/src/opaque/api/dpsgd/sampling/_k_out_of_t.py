"""Global k-out-of-t random-allocation sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from opaque.random.types import RngKey
from opaque.sampling import Sampler

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sized


class KOutOfTSampler(Sampler[list[int]]):
    """Each example chooses exactly ``k`` of ``t`` steps uniformly.

    The streaming implementation uses O(dataset-size) state: at each step, a
    record with ``r`` remaining participations and ``u`` remaining steps is
    selected with probability ``r / u``.
    """

    def __init__(
        self,
        data_source: Sized,
        *,
        total_participations: int,
        n_steps: int,
        key: RngKey,
    ) -> None:
        if len(data_source) == 0:
            raise ValueError("data_source must not be empty")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if not 1 <= total_participations <= n_steps:
            raise ValueError(
                "total_participations must be in "
                f"[1, n_steps={n_steps}], got {total_participations}"
            )
        self.data_source: Sized = data_source
        self.total_participations = int(total_participations)
        self.n_steps = int(n_steps)
        self._num_samples = len(data_source)
        self._key = key
        self._consumed = 0

    @property
    def consumed(self) -> int:
        return self._consumed

    @property
    def expected_batch_size(self) -> float:
        return self._num_samples * self.total_participations / self.n_steps

    def _batches(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self._key.seed)
        remaining = np.full(
            self._num_samples, self.total_participations, dtype=np.int64
        )
        for step in range(self.n_steps):
            probabilities = remaining / (self.n_steps - step)
            mask = rng.random(self._num_samples) < probabilities
            yield np.flatnonzero(mask).astype(int).tolist()
            remaining -= mask

    def __iter__(self) -> Iterator[list[int]]:
        for step, batch in enumerate(self._batches()):
            if step < self._consumed:
                continue
            self._consumed = step + 1
            yield batch

    def __len__(self) -> int:
        return self.n_steps - self._consumed


def _state_dict_k_out_of_t(sampler: KOutOfTSampler) -> dict[str, Any]:
    return {
        "key_seed": int(sampler._key.seed),
        "key_impl": str(sampler._key.impl),
        "consumed": sampler.consumed,
        "num_samples": sampler._num_samples,
        "total_participations": sampler.total_participations,
        "n_steps": sampler.n_steps,
    }


def _from_state_dict_k_out_of_t(
    template: KOutOfTSampler,
    state: Mapping[str, Any],
) -> KOutOfTSampler:
    if len(template.data_source) != int(state["num_samples"]):
        raise ValueError(
            "KOutOfTSampler.from_state_dict: template dataset length "
            f"{len(template.data_source)} does not match snapshot "
            f"num_samples={state['num_samples']}"
        )
    restored = KOutOfTSampler(
        template.data_source,
        total_participations=int(state["total_participations"]),
        n_steps=int(state["n_steps"]),
        key=RngKey(seed=int(state["key_seed"]), impl=str(state["key_impl"])),
    )
    restored._consumed = int(state["consumed"])
    return restored


def _register_k_out_of_t_sampler_serializer() -> None:
    from opaque.serialization import register_serializer

    register_serializer(
        KOutOfTSampler,
        _state_dict_k_out_of_t,
        _from_state_dict_k_out_of_t,
    )


_register_k_out_of_t_sampler_serializer()

__all__ = ["KOutOfTSampler"]
