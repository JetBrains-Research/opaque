"""Portable, backend-dispatched array operations.

The exported callables are canonical :class:`~opaque.primitive.Primitive`
objects.  They accept and return native arrays and dtype objects of the active
backend; Opaque deliberately does not wrap either.
"""

from typing import Any

from opaque.api.engine.primitive import PrimitiveTier, primitive


@primitive(tier=PrimitiveTier.CORE)
def is_array(value: object) -> bool:
    """Return whether ``value`` is an array of the active backend.

    False for Python scalars, numpy arrays under a non-numpy backend, and
    every pytree container. Use it to skip non-array leaves when walking a
    pytree, which may legitimately hold metadata beside its arrays.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def dtype(value: object) -> object:
    """Return the native dtype object of an array.

    The result is the provider's own dtype object — pass it back to
    :func:`astype`, :func:`sum`, or a ``dtype=`` argument rather than comparing
    it against a hard-coded provider literal.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def shape(value: object) -> tuple[int, ...]:
    """Return an array's dimensions as a plain tuple of ints.

    Always a ``tuple``, never a provider shape object, so it is safe to
    compare, unpack, and slice. A 0-d array gives ``()``.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def is_floating(value: object) -> bool:
    """Return whether an array or dtype is real floating point.

    Accepts an array or a dtype. Complex dtypes are **not** floating here —
    test them with :func:`is_complex` — and integer and boolean dtypes are not
    either.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def is_low_precision(value: object) -> bool:
    """Return whether an array or dtype is a float narrower than 32 bits.

    Accepts an array or a dtype. True for the 16-bit floats (``float16``,
    ``bfloat16``); False for ``float32`` and wider, and for every non-float
    dtype. This is the guard for the one-line rule mechanisms use to pick a
    compute dtype::

        compute = ops.float32() if ops.is_low_precision(leaf) else ops.dtype(leaf)
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def is_complex(value: object) -> bool:
    """Return whether an array or dtype is complex. Accepts either."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def float32() -> object:
    """Return the provider's 32-bit floating-point dtype.

    The neutral spelling of a dtype literal. Use it wherever documentation or
    a mechanism argument calls for ``float32`` — writing a provider's own dtype
    object binds that code to one backend.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def float64() -> object:
    """Return the provider's 64-bit floating-point dtype.

    The widest float every provider must offer, and the one a ``compute_dtype``
    argument takes to raise a mechanism's precision above the default.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def boolean() -> object:
    """Return the provider's boolean dtype.

    Predicate results (:func:`greater`, :func:`isfinite`, :func:`all`) already
    carry it; this is how you *construct* one, e.g. a neutral element for a
    :func:`minimum` fold over per-leaf predicates.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def real_dtype(value: object) -> object:
    """Return the dtype of a complex dtype's components.

    Accepts an array or a dtype, and returns a real float dtype: ``complex64``
    gives ``float32``. A real dtype is returned unchanged, so this is safe to
    apply unconditionally.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def scalar(value: object, *, dtype: object = None, like: object = None) -> object:
    """Create a native 0-d array holding ``value``.

    Args:
        value: Python scalar to materialize.
        dtype: Native dtype for the result. Overrides ``like``'s dtype.
        like: Array whose device *and* dtype the result adopts. Passing a leaf
            is the portable way to place a constant beside it at its own
            precision — a bare ``scalar(1.0)`` lands at the provider's default
            float, which is narrower than an ``float64`` leaf.

    Returns:
        A native 0-d array. With neither ``dtype`` nor ``like``, the provider
        picks the dtype implied by ``value``'s Python type.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def zeros(shape: Any, *, dtype: object = None, like: object = None) -> object:
    """Create a native array of ``shape`` filled with zeros.

    Args:
        shape: Sequence of dimensions; ``()`` gives a 0-d array.
        dtype: Native dtype for the result. Overrides ``like``'s dtype.
        like: Array whose device *and* dtype the result adopts.

    Returns:
        A zero-filled native array. Use :func:`zeros_like` when the shape
        should come from the reference array too.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def zeros_like(value: object) -> object:
    """Create a zero-filled array with ``value``'s shape, dtype, and device."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def ones_like(value: object) -> object:
    """Create a one-filled array with ``value``'s shape, dtype, and device."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def astype(value: object, value_dtype: object) -> object:
    """Return ``value`` converted to ``value_dtype``.

    Always returns an array of ``value_dtype``; whether that is a copy or the
    input itself when the dtype already matches is a provider detail, so do not
    rely on either for aliasing.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def clone(value: object) -> object:
    """Return an independent copy of ``value`` that shares no storage with it."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def detach(value: object) -> object:
    """Return ``value`` cut out of the autodiff graph.

    The result shares storage with ``value`` where the provider allows it;
    compose with :func:`clone` when you need an independent buffer as well.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def transfer(value: object, *args: Any, **kwargs: Any) -> object:
    """Move or convert ``value``, forwarding arguments to the provider.

    The one deliberately provider-shaped operation in this module: placement
    vocabularies differ too much between backends to normalize. Every provider
    accepts ``device=`` and ``dtype=`` keywords; anything else (a positional
    device, a memory format) is that provider's own spelling and makes the call
    site backend-specific. Prefer :func:`astype` for a pure dtype change.

    Returns:
        A native array on the requested device and dtype.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def scalar_item(value: object) -> Any:
    """Extract a Python scalar from a single-element array.

    Accepts any array holding exactly one element, whatever its rank, and
    returns ``float``, ``int``, or ``bool`` to match its dtype. This is a
    device-to-host synchronization point; it raises if the array holds more
    than one element.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def sqrt(value: object) -> object:
    """Take the elementwise square root.

    A negative input gives NaN, not a complex result. Integer input is
    promoted to a float.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def exp(value: object) -> object:
    """Take the elementwise natural exponential.

    Overflows to ``+inf`` rather than raising, and does so far sooner in a
    low-precision dtype.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def erf(value: object) -> object:
    """Evaluate the Gauss error function elementwise.

    Maps the reals onto ``(-1, 1)``; ``(1 + erf(z / sqrt(2))) / 2`` is the
    standard normal CDF, which is how a bounded mechanism turns a normal draw
    into a uniform one.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def erfinv(value: object) -> object:
    """Invert the Gauss error function elementwise.

    Defined on ``(-1, 1)``: ``±1`` gives ``±inf`` and anything outside gives
    NaN. Clamp the input away from the endpoints by :func:`finfo_eps` before
    calling when it comes from a computed probability.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def finfo_eps(value_dtype: object) -> float:
    """Return the machine epsilon of a floating dtype, as a Python float.

    Accepts an array or a dtype. Epsilon is the gap between 1.0 and the next
    representable value — the relative rounding unit that sets how far a
    computed bound must be shrunk to stay a bound once it is stored.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def finfo_smallest_normal(value_dtype: object) -> float:
    """Return a floating dtype's smallest positive normal, as a Python float.

    Accepts an array or a dtype. Below this magnitude a value is subnormal and
    loses relative precision, so it bounds the *absolute* error term that
    :func:`finfo_eps`'s relative one cannot cover.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def to_host(value: object) -> object:
    """Return a detached host-memory ``numpy.ndarray`` copy of a native array.

    The one sanctioned exit from provider-native arrays into host numpy —
    for scores, metrics, and other results that leave the compute graph.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def rsqrt(value: object) -> object:
    """Take the elementwise reciprocal square root, ``1 / sqrt(value)``.

    Zero gives ``+inf`` and a negative input gives NaN. Adding a small floor to
    the input is the usual guard where the result scales a gradient.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def square(value: object) -> object:
    """Square each value, preserving the input dtype.

    On a complex array this squares the value, not its modulus; use
    :func:`abs` first for ``|z|²``.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def abs(value: object) -> object:
    """Take the elementwise absolute value.

    On a complex array this is the modulus, so the result is real-valued.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def add(left: object, right: object) -> object:
    """Add elementwise, broadcasting operands and promoting their dtypes."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def subtract(left: object, right: object) -> object:
    """Subtract ``right`` from ``left`` elementwise, broadcasting operands."""
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def multiply(left: object, right: object) -> object:
    """Multiply elementwise, broadcasting operands and promoting their dtypes.

    Either operand may be a Python scalar, which is how a stddev or a clip
    scale is applied to an array without materializing one first.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def divide(left: object, right: object) -> object:
    """Divide elementwise, broadcasting operands.

    Always true division: integer operands give a float result. Division by
    zero follows IEEE — ``±inf``, or NaN for ``0/0`` — rather than raising, so
    guard the divisor or sanitize with :func:`nan_to_num`.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def pow(value: object, exponent: object) -> object:
    """Raise ``value`` to ``exponent`` elementwise, broadcasting operands.

    A negative base with a non-integer exponent gives NaN, not a complex
    result.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def mean(value: object, axis: Any = None) -> object:
    """Average array values along an optional axis.

    ``axis=None`` reduces every axis to a 0-d array. The input must be a float
    or complex array — providers reject integer input rather than guessing an
    output dtype, so cast with :func:`astype` first.

    Returns:
        A native array, never a Python number.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def reciprocal(value: object) -> object:
    """Take the elementwise reciprocal, ``1 / value``.

    Zero gives ``±inf`` following IEEE rather than raising.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def sum(value: object, axis: Any = None, dtype: object = None) -> object:
    """Sum array values along an optional axis.

    Args:
        value: Array to reduce.
        axis: Axis or axes to reduce. ``None`` reduces every axis to a 0-d
            array; negative axes count from the end.
        dtype: Accumulation *and* result dtype. Setting it wider than the
            input is how a low-precision sum avoids losing terms — see
            :func:`accumulator_dtype`.

    Returns:
        A native array, never a Python number: a full reduction gives a 0-d
        array. Call :func:`scalar_item` when you need a Python value.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def accumulator_dtype(value: object, *, kind: str = "sum") -> object:
    """Return the dtype to accumulate ``value`` in on its own device.

    Wider than the input where the device supports it — ``float64`` on hosts
    and accelerators that implement it, ``float32`` where it is unavailable or
    prohibitively slow — so the answer is device-dependent by design and must
    be asked for per array rather than assumed.

    Args:
        value: Array whose accumulation dtype is wanted; its device decides.
        kind: The reduction being accumulated. ``"sum"`` and ``"mean"`` are
            defined; providers may recognize more.

    Returns:
        A native dtype to pass to :func:`sum` or :func:`astype`.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def amin(value: object, axis: Any = None) -> object:
    """Reduce to the smallest value along an optional axis.

    ``axis=None`` reduces every axis to a 0-d array. Distinct from
    :func:`minimum`, which compares two arrays elementwise and keeps their
    shape. NaN propagates: a reduction over any NaN gives NaN.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def amax(value: object, axis: Any = None) -> object:
    """Reduce to the largest value along an optional axis.

    ``axis=None`` reduces every axis to a 0-d array. Distinct from
    :func:`maximum`, which compares two arrays elementwise and keeps their
    shape. NaN propagates.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def greater(left: object, right: object) -> object:
    """Test ``left > right`` elementwise, broadcasting operands.

    Returns a boolean array. NaN compares false against everything, itself
    included.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def minimum(left: object, right: object) -> object:
    """Take the smaller of two values elementwise, broadcasting operands.

    On boolean arrays this is a logical AND — the portable spelling for
    folding per-leaf predicates together, since it stays an array and survives
    :func:`~opaque.autodiff.vmap`.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def maximum(left: object, right: object) -> object:
    """Take the larger of two values elementwise, broadcasting operands.

    On boolean arrays this is a logical OR.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def where(condition: object, left: object, right: object) -> object:
    """Choose from ``left`` where ``condition`` is true, else ``right``.

    All three operands broadcast against each other, and ``left`` / ``right``
    may be Python scalars. Both branches are evaluated, so this guards a
    *value*, not a computation: use it to replace a nonfinite result, not to
    avoid producing one.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def isfinite(value: object) -> object:
    """Test each value for being neither NaN nor infinite.

    Returns a boolean array of the input's shape. Integer and boolean inputs
    are finite everywhere.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def all(value: object, axis: Any = None) -> object:
    """Test whether every value along an optional axis is true.

    ``axis=None`` reduces every axis to a 0-d array. Non-boolean input is
    tested for nonzero-ness.

    Returns:
        A boolean native array, never a Python ``bool``. Keeping it an array is
        what lets a predicate survive :func:`~opaque.autodiff.vmap` and compose
        with :func:`minimum`; call :func:`scalar_item` to leave the graph.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def nan_to_num(
    value: object,
    *,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
) -> object:
    """Replace NaN and infinities with finite substitutes.

    The defaults substitute **zero for all three**, which is deliberate and
    differs from NumPy and Torch: those saturate infinities to the dtype's
    largest finite values. In a DP mechanism a nonfinite contribution is a
    failed record, and zero is the only substitute that leaves the sensitivity
    bound intact — saturating would hand the aggregate the largest value the
    dtype can hold.

    Args:
        value: Array to sanitize.
        nan: Substitute for NaN.
        posinf: Substitute for ``+inf``.
        neginf: Substitute for ``-inf``.

    Returns:
        An array of the input's shape and dtype with no nonfinite values.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def clamp(value: object, lo: object = None, hi: object = None) -> object:
    """Bound array values from below by ``lo`` and above by ``hi``.

    Either bound may be omitted to leave that side unbounded, but at least one
    must be given. Bounds may be Python scalars or broadcastable arrays.

    NaN is **not** a value clamping can bound, and it propagates unchanged.
    Sanitize first with :func:`nan_to_num` when the result feeds a sensitivity
    bound.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def concatenate(values: Any, axis: int = 0) -> object:
    """Join arrays along an existing axis.

    Args:
        values: Any iterable of arrays that agree on every axis but ``axis``.
        axis: Axis to join along; negative counts from the end.

    Returns:
        One array whose ``axis`` length is the sum of the inputs'. Every input
        keeps its rank — concatenating 1-d arrays gives a 1-d array.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def slice_array(value: object, slices: Any) -> object:
    """Index ``value``, as the subscript operator would.

    Args:
        value: Array to index.
        slices: What you would write inside ``[]``. An ``int`` selects along
            the first axis and drops it; a ``slice`` keeps the rank; a
            **tuple** indexes successive axes, so ``(0, 1)`` on a 2-d array
            gives a 0-d array rather than two rows.

    Returns:
        A view or copy — providers differ, so do not write through the result.
        Advanced indexing (boolean masks, index arrays) is provider-specific
        and outside this contract.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def expand_dims(value: object, axis: int) -> object:
    """Insert a size-one dimension at ``axis``.

    ``axis`` is a position in the *result*, and may be negative to count from
    its end: ``-1`` appends a trailing axis.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def squeeze(value: object, axis: int | None = None) -> object:
    """Remove size-one dimensions.

    ``axis=None`` removes every size-one dimension. Naming an ``axis`` removes
    only that one, and — following the providers rather than NumPy — leaves the
    array unchanged when that axis is not size one, rather than raising. Check
    :func:`shape` yourself if the distinction matters.
    """
    raise NotImplementedError


@primitive(tier=PrimitiveTier.CORE)
def promote_dtype(first: object, second: object) -> object:
    """Return the dtype an operation on both operands would produce.

    Accepts arrays or dtypes in either position. Use it to pick one dtype for a
    computation up front, rather than discovering the provider's promotion
    after the fact.
    """
    raise NotImplementedError


__all__ = [
    "abs",
    "accumulator_dtype",
    "add",
    "amax",
    "amin",
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
    "boolean",
    "float32",
    "float64",
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
    "to_host",
    "transfer",
    "where",
    "zeros",
    "zeros_like",
]
