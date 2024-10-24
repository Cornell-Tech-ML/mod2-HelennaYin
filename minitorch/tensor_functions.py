"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Union

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple, Optional

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Union[Tensor, int, Tuple[int, ...], None]) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        tensor_vals = []
        for v in vals:
            if isinstance(v, minitorch.Tensor):  # Only process Tensor objects
                raw_vals.append(v.detach())  # Get the raw value without history
                tensor_vals.append(v)  # Track the tensor
                if v.requires_grad():  # Check if gradient is needed
                    need_grad = True
            else:
                raw_vals.append(
                    v
                )  # For non-tensor values (like dim or atol), pass as-is

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(
            c, minitorch.Tensor
        ), f"Expected return type Tensor, got {type(c)}"
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, tensor_vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward method for negation"""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Borward method for negation"""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward method for inverse"""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Borward method for inverse"""
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward method for add"""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Borward method for add"""
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[int] = None) -> Tensor:
        """Return 1 if all are true"""
        # mapped = SimpleBackend.map(lambda x: 1.0 if x == 1.0 else 0.0)(a)

        # Step 2: Reduce by multiplying along the specified dimension
        # If dim is None, reduce across all elements to a single value
        if dim is not None:
            reduced = a.f.mul_reduce(a, dim)
        else:
            reduced = tensor([1])

        # The result will be 1.0 if all elements were 1.0, else 0.0
        return reduced


class Mul(Function):
    """Multiplies two tensors element-wise."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward method for mul"""
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Borward method for mul"""
        t1, t2 = ctx.saved_values
        grad_t1 = grad_output.f.mul_zip(t2, grad_output)
        grad_t2 = grad_output.f.mul_zip(t1, grad_output)
        return grad_t1, grad_t2


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tuple[int, ...]) -> Tensor:
        """Forward method for view"""
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(len(shape))]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor,]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
        )


class Sum(Function):
    """Sums elements of the tensor along the specified dimension."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Optional[int] = None) -> Tensor:
        """Forward method for sum"""
        ctx.save_for_backward(t1, dim)
        if dim is not None:
            reduced = t1.f.add_reduce(t1, dim)
        else:
            reduced = t1.f.add_reduce(
                t1.contiguous().view(int(operators.prod(list(t1.shape)))), 0
            )
        return reduced

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Borward method for sum"""
        (t1, dim) = ctx.saved_values
        shape = t1.shape
        if dim is None:
            out = grad_output.zeros(shape)
            out._tensor._storage[:] = grad_output[0]
            return out
        else:
            return grad_output


class Sigmoid(Function):
    """Applies the sigmoid function element-wise."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward method for sigmoid"""
        ctx.save_for_backward(t1)
        return t1.f.sigmoid_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Borward method for sigmoid"""
        (t1,) = ctx.saved_values
        return t1.f.add_zip(
            grad_output.f.mul_zip(grad_output, t1.f.sigmoid_map(t1)),
            grad_output.f.neg_map(
                grad_output.f.mul_zip(
                    grad_output,
                    t1.f.mul_zip(t1.f.sigmoid_map(t1), t1.f.sigmoid_map(t1)),
                )
            ),
        )


class ReLU(Function):
    """Applies the rectified linear unit (ReLU) function element-wise."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward method for relu"""
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Borward method for relu"""
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class Log(Function):
    """Applies the logarithm function element-wise."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward method for log"""
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Borward method for log"""
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    """Applies the exponential function element-wise."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Forward method for log"""
        ctx.save_for_backward(t1)
        return t1.f.exp_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Borward method for log"""
        (t1,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, t1.f.exp_map(t1))


class LT(Function):
    """Performs element-wise less-than comparison between two tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward method for less than"""
        ctx.save_for_backward(t1, t2)
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Borward method for less than"""
        zero_grad = grad_output.zeros(ctx.saved_values[0].shape)
        return zero_grad, zero_grad


class EQ(Function):
    """Performs element-wise equality comparison between two tensors."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Forward method for equal"""
        ctx.save_for_backward(t1, t2)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Borward method for equal"""
        zero_grad = grad_output.zeros(ctx.saved_values[0].shape)
        return zero_grad, zero_grad


class IsClose(Function):
    """Checks if two tensors are element-wise close within a tolerance."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor, atol: float = 1e-5) -> Tensor:
        """Forward method for isclose"""
        return t1.f.is_close_zip(t1, t2)


class Permute(Function):
    """Permutes the dimensions of the tensor."""

    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tuple[int]) -> Tensor:
        """Forward method for permute"""
        ctx.save_for_backward(t1, order)
        tens_store = t1._tensor.permute(*order)
        return minitorch.Tensor.make(
            tens_store._storage,
            tens_store.shape,
            tens_store.strides,
            backend=t1.backend,
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Borward method for permute"""
        t1, order = ctx.saved_values
        return minitorch.Tensor.make(
            grad_output._tensor._storage,
            t1._tensor.shape,
            t1._tensor.strides,
            backend=grad_output.backend,
        )


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(list(shape))), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(list(shape))))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute central difference gradient for `f` at `x` with respect to `arg`."""
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
