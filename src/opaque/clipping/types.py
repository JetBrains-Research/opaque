"""Type definitions for clipping operations."""

from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

AuxiliaryOutput = namedtuple("AuxiliaryOutput", ["values", "grad_norms", "aux"])


@dataclass(frozen=True)
class BoundedSensitivityCallable:
    """Callable with a sensitivity property.

    If has_aux is False, the sensitivity guarantee holds for the entire output
    which may be an arbitrary pytree of Tensors. If has_aux is True, the
    output of the function is a pair `(value, aux)` and the sensitivity guarantee
    only holds for `value` PyTree. The aux PyTree is returned on a per-example
    basis (i.e., as a PyTree of tensors having a batch axis). The caller should
    handle the aux output with care w.r.t. DP guarantees, should they be needed.
    """

    fun: Callable[..., Any]
    l2_norm_bound: float
    has_aux: bool

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)

    def sensitivity(self, neighboring_relation: str = "REPLACE_SPECIAL") -> float:
        """Returns the L2 sensitivity of the Callable.

        The L2 sensitivity is defined with respect to the given neighboring relation
        and the unit of privacy implied by the function that created this instance.

        Args:
            neighboring_relation: The neighboring relation to consider. One of:
                - "ADD_OR_REMOVE_ONE": Dataset differs by adding/removing one record
                - "REPLACE_ONE": Dataset differs by replacing one record
                - "REPLACE_SPECIAL": Dataset differs by replacing one record with a special element

        Returns:
            The L2 sensitivity of the Callable.

        Raises:
            ValueError: If neighboring_relation is not supported.
        """
        if neighboring_relation == "ADD_OR_REMOVE_ONE":
            return self.l2_norm_bound
        elif neighboring_relation == "REPLACE_ONE":
            return 2 * self.l2_norm_bound
        elif neighboring_relation == "REPLACE_SPECIAL":
            return self.l2_norm_bound
        else:
            raise ValueError(f"Unsupported neighboring_relation={neighboring_relation}")


__all__ = ["AuxiliaryOutput", "BoundedSensitivityCallable"]
