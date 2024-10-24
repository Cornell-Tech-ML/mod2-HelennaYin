from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the forward pass to the given Scalar or numeric values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the forward pass for addition."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Performs the backward pass for addition."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass for log."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Performs the backward pass for log."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    """Multplication function $f(x) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the forward pass for multiplication."""
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Performs the backward pass for multiplication."""
        (
            a,
            b,
        ) = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass for inverse."""
        ctx.save_for_backward(a)
        return 1.0 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Performs the backward pass for inverse."""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass for neg."""
        ctx.save_for_backward(a)
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Performs the backward pass for neg."""
        (a,) = ctx.saved_values
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass for sigmoid."""
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Performs the backward pass for sigmoid."""
        (a,) = ctx.saved_values
        sig = operators.sigmoid(a)
        return d_output * sig * (1 - sig)


class ReLU(ScalarFunction):
    """Relu function: 0 if x <0, x if x >0"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass for ReLU."""
        ctx.save_for_backward(a)
        return float(max(0, a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Performs the backward pass for ReLU."""
        (a,) = ctx.saved_values
        return float(operators.relu_back(a, d_output))


class Exp(ScalarFunction):
    """Exponential function: e^x"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Performs the forward pass for exponential function."""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Performs the forward pass for exponential function."""
        (a,) = ctx.saved_values
        exp_a = operators.exp(a)
        return d_output * exp_a


class LT(ScalarFunction):
    """Less than function: True if x < y, False otherwise"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the forward pass for less than."""
        ctx.save_for_backward(a, b)
        return float(a < b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Performs the backward pass for less than."""
        return 0, 0


class EQ(ScalarFunction):
    """Equal function: True if x == y, False otherwise"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Performs the forward pass for equal to."""
        ctx.save_for_backward(a, b)
        return float(a == b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Performs the backward pass for equal to."""
        return 0, 0
