"""Portable, backend-dispatched array operations.

The exported callables are canonical :class:`~opaque.primitive.Primitive`
objects.  They accept and return native arrays and dtype objects of the active
backend; Opaque deliberately does not wrap either.
"""

from opaque.api.engine.primitive import Primitive


def _core(name: str) -> Primitive:
    return Primitive(f"opaque.ops.{name}", tier="core")


is_array = _core("is_array")
dtype = _core("dtype")
shape = _core("shape")
is_floating = _core("is_floating")
is_low_precision = _core("is_low_precision")
is_complex = _core("is_complex")
float32 = _core("float32")
real_dtype = _core("real_dtype")
scalar = _core("scalar")
zeros = _core("zeros")
zeros_like = _core("zeros_like")
ones_like = _core("ones_like")
astype = _core("astype")
clone = _core("clone")
detach = _core("detach")
transfer = _core("transfer")
scalar_item = _core("scalar_item")
sqrt = _core("sqrt")
square = _core("square")
abs = _core("abs")
add = _core("add")
subtract = _core("subtract")
multiply = _core("multiply")
divide = _core("divide")
sum = _core("sum")
greater = _core("greater")
minimum = _core("minimum")
maximum = _core("maximum")
where = _core("where")
isfinite = _core("isfinite")
all = _core("all")
nan_to_num = _core("nan_to_num")
clamp = _core("clamp")
concatenate = _core("concatenate")
slice_array = _core("slice_array")
promote_dtype = _core("promote_dtype")

__all__ = [
    "abs",
    "add",
    "all",
    "astype",
    "clamp",
    "clone",
    "concatenate",
    "detach",
    "divide",
    "dtype",
    "float32",
    "greater",
    "is_array",
    "is_complex",
    "is_floating",
    "is_low_precision",
    "isfinite",
    "maximum",
    "minimum",
    "multiply",
    "nan_to_num",
    "ones_like",
    "promote_dtype",
    "real_dtype",
    "scalar",
    "scalar_item",
    "shape",
    "slice_array",
    "sqrt",
    "square",
    "subtract",
    "sum",
    "transfer",
    "where",
    "zeros",
    "zeros_like",
]
