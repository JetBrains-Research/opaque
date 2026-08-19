"""Portable, backend-dispatched array operations.

The exported callables are canonical :class:`~opaque.primitive.Primitive`
objects.  They accept and return native arrays and dtype objects of the active
backend; Opaque deliberately does not wrap either.
"""

from typing import Any

from opaque.api.engine.primitive import PrimitiveTier, primitive


@primitive(tier=PrimitiveTier.CORE)
def is_array(value: object) -> bool:
    """Return whether ``value`` is a native array."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def dtype(value: object) -> object:
    """Return the native dtype of ``value``."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def shape(value: object) -> tuple[int, ...]:
    """Return the dimensions of ``value``."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def is_floating(value: object) -> bool:
    """Return whether an array or dtype is floating point."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def is_low_precision(value: object) -> bool:
    """Return whether an array or dtype uses low precision."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def is_complex(value: object) -> bool:
    """Return whether an array or dtype is complex."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def float32() -> object:
    """Return the provider's 32-bit floating-point dtype."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def real_dtype(value: object) -> object:
    """Return the real component dtype for an array or dtype."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def scalar(value: object, *, dtype: object = None, like: object = None) -> object:
    """Create a native scalar array."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def zeros(shape: Any, *, dtype: object = None, like: object = None) -> object:
    """Create a native array filled with zeros."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def zeros_like(value: object) -> object:
    """Create a zero-filled array like ``value``."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def ones_like(value: object) -> object:
    """Create a one-filled array like ``value``."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def astype(value: object, value_dtype: object) -> object:
    """Convert ``value`` to a native dtype."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def clone(value: object) -> object:
    """Clone a native array."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def detach(value: object) -> object:
    """Detach a native array from automatic differentiation."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def transfer(value: object, *args: Any, **kwargs: Any) -> object:
    """Transfer a native array according to provider arguments."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def scalar_item(value: object) -> Any:
    """Extract a Python scalar from a scalar array."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def sqrt(value: object) -> object:
    """Compute an elementwise square root."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def exp(value: object) -> object:
    """Compute an elementwise natural exponential."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def erf(value: object) -> object:
    """Compute the elementwise Gauss error function."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def erfinv(value: object) -> object:
    """Compute the elementwise inverse Gauss error function."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def finfo_eps(value_dtype: object) -> float:
    """Return the machine epsilon of a native floating-point dtype."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def finfo_smallest_normal(value_dtype: object) -> float:
    """Return the smallest positive normal value of a native floating dtype."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def rsqrt(value: object) -> object:
    """Compute an elementwise reciprocal square root."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def square(value: object) -> object:
    """Compute an elementwise square."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def abs(value: object) -> object:
    """Compute an elementwise absolute value."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def add(left: object, right: object) -> object:
    """Add two values elementwise."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def subtract(left: object, right: object) -> object:
    """Subtract two values elementwise."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def multiply(left: object, right: object) -> object:
    """Multiply two values elementwise."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def divide(left: object, right: object) -> object:
    """Divide two values elementwise."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def pow(value: object, exponent: object) -> object:
    """Raise ``value`` to ``exponent`` elementwise."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def mean(value: object, axis: Any = None) -> object:
    """Compute the mean of array values along an optional axis."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def reciprocal(value: object) -> object:
    """Compute an elementwise reciprocal."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def sum(value: object, axis: Any = None, dtype: object = None) -> object:
    """Sum array values along an optional axis."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def accumulator_dtype(value: object, *, kind: str = "sum") -> object:
    """Return a dtype suitable for stable accumulation on the active backend."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def greater(left: object, right: object) -> object:
    """Compare two values elementwise."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def minimum(left: object, right: object) -> object:
    """Compute the elementwise minimum of two values."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def maximum(left: object, right: object) -> object:
    """Compute the elementwise maximum of two values."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def where(condition: object, left: object, right: object) -> object:
    """Select values elementwise according to ``condition``."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def isfinite(value: object) -> object:
    """Test array values for finiteness."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def all(value: object, axis: Any = None) -> object:
    """Reduce boolean values along an optional axis."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def nan_to_num(value: object) -> object:
    """Replace nonfinite array values with finite values."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def clamp(value: object, lo: object = None, hi: object = None) -> object:
    """Clamp array values to optional bounds."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def concatenate(values: Any, axis: int = 0) -> object:
    """Concatenate arrays along an axis."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def slice_array(value: object, slices: Any) -> object:
    """Index an array with provider-native slicing semantics."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def expand_dims(value: object, axis: int) -> object:
    """Insert a size-one dimension at ``axis``."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def squeeze(value: object, axis: int | None = None) -> object:
    """Remove size-one dimensions, optionally only at ``axis``."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def promote_dtype(first: object, second: object) -> object:
    """Return the dtype produced by promoting two dtypes."""
    raise NotImplementedError


__all__ = [
    "abs",
    "accumulator_dtype",
    "add",
    "all",
    "astype",
    "clamp",
    "clone",
    "concatenate",
    "detach",
    "divide",
    "dtype",
    "erf",
    "erfinv",
    "exp",
    "expand_dims",
    "finfo_eps",
    "finfo_smallest_normal",
    "float32",
    "greater",
    "is_array",
    "is_complex",
    "is_floating",
    "is_low_precision",
    "isfinite",
    "maximum",
    "mean",
    "minimum",
    "multiply",
    "nan_to_num",
    "ones_like",
    "pow",
    "promote_dtype",
    "real_dtype",
    "reciprocal",
    "rsqrt",
    "scalar",
    "scalar_item",
    "shape",
    "slice_array",
    "sqrt",
    "square",
    "squeeze",
    "subtract",
    "sum",
    "transfer",
    "where",
    "zeros",
    "zeros_like",
]
