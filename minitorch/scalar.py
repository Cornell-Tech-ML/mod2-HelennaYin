from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Returns True if this variable has no history and is a constant."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the input variables used in the operation that created this Scalar."""
        assert self.history is not None
        return self.history.inputs

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object."""
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Implements the chain rule to propagate gradients to inputs.

        Args:
        ----
            d_output: The gradient from the output.

        Returns:
        -------
            An iterable of tuples where each tuple contains a parent Variable and its gradient.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        fn = h.last_fn
        inputs = h.inputs
        ctx = h.ctx

        local_gradients = fn._backward(ctx, d_output)

        if len(inputs) == 1:
            if not isinstance(local_gradients, (tuple, list)):
                local_gradients = (local_gradients,)

        for input_var, local_grad in zip(inputs, local_gradients):
            if not input_var.is_constant():
                yield input_var, local_grad

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Compares whether this Scalar is less than another value."""
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Compares whether this Scalar is greater than another value."""
        return LT.apply(b, self)

    def __neg__(self) -> Scalar:
        """Negates the Scalar."""
        return Neg.apply(self)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Subtracts another Scalar or value from this Scalar."""
        return self + (-b)

    def __add__(self, b: ScalarLike) -> Scalar:
        """Adds this Scalar to another Scalar or value."""
        return Add.apply(self, b)

    def __eq__(self, b: ScalarLike) -> Scalar:
        """Checks whether this Scalar is equal to another Scalar or value."""
        return EQ.apply(self, b)

    def log(self) -> Scalar:
        """Computes the logarithm of the Scalar."""
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Computes the exponential of the Scalar."""
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Computes the sigmoid of the Scalar."""
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Applies the ReLU activation function to the Scalar."""
        return ReLU.apply(self)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that automatic differentiation works correctly on a Python function.
    It compares the computed derivatives using backpropagation with numerical derivatives
    computed using central difference.

    Args:
    ----
        f (Any): A function from n-scalars to 1-scalar for which the derivative is to be checked.
        *scalars (Scalar): The input scalar values for the function `f` whose derivatives are to be tested.

    Returns:
    -------
        None: The function asserts that the computed derivatives match the numerical derivatives.

    Raises:
    ------
        AssertionError: If the computed derivative does not match the central difference approximation.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
