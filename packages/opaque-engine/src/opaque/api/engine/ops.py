"""Portable, backend-dispatched array operations.

The exported callables are canonical :class:`~opaque.primitive.Primitive`
objects.  They accept and return native arrays and dtype objects of the active
backend; Opaque deliberately does not wrap either.

Operations required of every provider are declared at
:attr:`~opaque.primitive.PrimitiveTier.CORE`.  Operations that only some
mechanism families need — matrix construction, cumulative scans, dense linear
algebra, and real FFTs — are declared at
:attr:`~opaque.primitive.PrimitiveTier.OPTIONAL` and grouped into the named
:class:`ArrayProfile` profiles, so a provider missing them still activates and
callers can check :func:`supports_profile` before use.
"""

import builtins
from enum import StrEnum
from typing import Any

from opaque.api.engine.primitive import Primitive, PrimitiveTier, primitive


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
def zeros(value_shape: Any, *, dtype: object = None, like: object = None) -> object:
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


# ---------------------------------------------------------------------------
# Extended array construction and layout (ArrayProfile.EXTENDED)
# ---------------------------------------------------------------------------


@primitive(tier=PrimitiveTier.OPTIONAL)
def asarray(value: object, *, dtype: object = None, like: object = None) -> object:
    """Convert host data (sequences, NumPy arrays) to a native array."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def arange(
    start: object,
    stop: object = None,
    step: object = 1,
    *,
    dtype: object = None,
    like: object = None,
) -> object:
    """Create a native array of evenly spaced values."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def ones(value_shape: Any, *, dtype: object = None, like: object = None) -> object:
    """Create a native array filled with ones."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def eye(n: int, *, dtype: object = None, like: object = None) -> object:
    """Create an ``n x n`` identity matrix."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def diag(value: object) -> object:
    """Extract the diagonal of a matrix, or build a matrix from a vector."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def tril(value: object, k: int = 0) -> object:
    """Zero the entries strictly above the ``k``-th diagonal."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def reshape(value: object, value_shape: Any) -> object:
    """Return ``value`` with a new shape."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def transpose(value: object, axes: Any = None) -> object:
    """Permute array dimensions, reversing them when ``axes`` is ``None``."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def stack(values: Any, axis: int = 0) -> object:
    """Join arrays along a new axis."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def flip(value: object, axis: int) -> object:
    """Reverse array order along ``axis``."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def roll(value: object, shift: int, axis: int) -> object:
    """Shift array values along ``axis``, wrapping around the end."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def real(value: object) -> object:
    """Return the real component of a possibly complex array."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def log(value: object) -> object:
    """Compute an elementwise natural logarithm."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Scans, reductions, and ordering (ArrayProfile.SCAN)
# ---------------------------------------------------------------------------


@primitive(tier=PrimitiveTier.OPTIONAL)
def cumsum(value: object, axis: int = 0) -> object:
    """Compute a cumulative sum along ``axis``."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def cumprod(value: object, axis: int = 0) -> object:
    """Compute a cumulative product along ``axis``."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def cummax(value: object, axis: int = 0) -> object:
    """Compute a cumulative maximum along ``axis``."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def prod(value: object, axis: Any = None) -> object:
    """Multiply array values along an optional axis."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def amax(value: object, axis: Any = None) -> object:
    """Reduce array values to their maximum along an optional axis."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def amin(value: object, axis: Any = None) -> object:
    """Reduce array values to their minimum along an optional axis."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def any(value: object, axis: Any = None) -> object:
    """Reduce boolean values disjunctively along an optional axis."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def argmax(value: object, axis: Any = None) -> object:
    """Return the index of the maximum value along an optional axis."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def argsort(value: object, *, descending: bool = False) -> object:
    """Return the indices that sort a one-dimensional array."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def nonzero(value: object) -> object:
    """Return the flat indices of the nonzero entries of a 1-D array."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Dense linear algebra (ArrayProfile.LINALG)
# ---------------------------------------------------------------------------


@primitive(tier=PrimitiveTier.OPTIONAL)
def matmul(left: object, right: object) -> object:
    """Compute a matrix product."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def tensordot(left: object, right: object, axes: Any = 1) -> object:
    """Contract two arrays over the given axes."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def linalg_inv(value: object) -> object:
    """Invert a square matrix."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def linalg_eigvals(value: object) -> object:
    """Return the eigenvalues of a general square matrix."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Real-input FFTs (ArrayProfile.SPECTRAL)
# ---------------------------------------------------------------------------


@primitive(tier=PrimitiveTier.OPTIONAL)
def fft_rfft(value: object, n: int | None = None) -> object:
    """Compute the one-dimensional FFT of a real-valued array."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.OPTIONAL)
def fft_irfft(value: object, n: int | None = None) -> object:
    """Compute the inverse of :func:`fft_rfft`."""
    raise NotImplementedError


class ArrayProfile(StrEnum):
    """Named optional array-operation profiles."""

    EXTENDED = "extended"
    SCAN = "scan"
    LINALG = "linalg"
    SPECTRAL = "spectral"

    @property
    def primitives(self) -> tuple[Primitive, ...]:
        """Return the primitive declarations required by this profile."""
        return profile_primitives(self)

    def supports(self, backend: object | str | None = None) -> bool:
        """Return whether ``backend`` implements this complete profile."""
        return supports_profile(self, backend)


ARRAY_PROFILE_VERSION = 1
"""Version of the named optional array-operation profile contract."""


_ARRAY_PROFILES: dict[ArrayProfile, tuple[Primitive, ...]] = {
    ArrayProfile.EXTENDED: (
        asarray,
        arange,
        ones,
        eye,
        diag,
        tril,
        reshape,
        transpose,
        stack,
        flip,
        roll,
        real,
        log,
    ),
    ArrayProfile.SCAN: (
        cumsum,
        cumprod,
        cummax,
        prod,
        amax,
        amin,
        any,
        argmax,
        argsort,
        nonzero,
    ),
    ArrayProfile.LINALG: (
        matmul,
        tensordot,
        linalg_inv,
        linalg_eigvals,
    ),
    ArrayProfile.SPECTRAL: (
        fft_rfft,
        fft_irfft,
    ),
}


def profile_primitives(profile: ArrayProfile | str) -> tuple[Primitive, ...]:
    """Return the declarations required by a named array profile."""
    return _ARRAY_PROFILES[ArrayProfile(profile)]


def supports_profile(
    profile: ArrayProfile | str,
    backend: object | str | None = None,
) -> bool:
    """Return whether ``backend`` registered every primitive in ``profile``."""
    # ``all`` is shadowed by this module's elementwise reduction primitive.
    return builtins.all(
        operation.supports(backend) for operation in profile_primitives(profile)
    )


__all__ = [
    "ARRAY_PROFILE_VERSION",
    "ArrayProfile",
    "amax",
    "amin",
    "any",
    "arange",
    "argmax",
    "argsort",
    "asarray",
    "cummax",
    "cumprod",
    "cumsum",
    "diag",
    "eye",
    "fft_irfft",
    "fft_rfft",
    "flip",
    "linalg_eigvals",
    "linalg_inv",
    "log",
    "matmul",
    "nonzero",
    "ones",
    "prod",
    "profile_primitives",
    "real",
    "reshape",
    "roll",
    "stack",
    "supports_profile",
    "tensordot",
    "transpose",
    "tril",
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
