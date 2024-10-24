from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_plus = list(vals)
    vals_minus = list(vals)
    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon

    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)

    return (f_plus - f_minus) / (2 * epsilon)


class Variable(Protocol):
    """A protocol that defines the operations for autodiff variables."""

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative for this variable.

        Args:
        ----
            x : The derivative value to accumulate.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns the unique identifier of the variable."""
        ...

    def is_leaf(self) -> bool:
        """Checks if the variable is a leaf node (created by the user).

        Returns
        -------
            bool: True if the variable is a leaf, False otherwise.

        """
        ...

    def is_constant(self) -> bool:
        """Checks if the variable is constant.

        Returns
        -------
            bool: True if the variable is constant, False otherwise.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of this variable.

        Returns
        -------
            Iterable[Variable]: An iterable of parent variables.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Performs the chain rule to propagate gradients to parent variables.

        Args:
        ----
            d_output : The gradient from the output.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of parent variables and their gradients.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    topo_order = []

    def dfs(var: Variable) -> None:
        """Depth-first search (DFS) to visit each node.

        Args:
        ----
            var : The variable to explore.

        """
        if var.unique_id not in visited and not var.is_constant():
            visited.add(var.unique_id)
            for parent in var.parents:
                dfs(parent)
            topo_order.append(var)

    dfs(variable)

    topo_order.reverse()
    return topo_order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable

    deriv:
    -----
        Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    gradients = {variable.unique_id: deriv}

    # Mapping unique_id back to the original Scalar object for derivative accumulation
    id_to_variable = {variable.unique_id: variable}

    # Perform topological sorting of the computation graph to process nodes in the correct order
    topo_order = topological_sort(variable)

    # Iterate over the variables in reverse topological order
    for var in topo_order:
        # Get the current derivative for this variable
        d_output = gradients.get(var.unique_id, 0.0)

        # Ensure that d_output is a float, not a tuple
        if isinstance(d_output, (tuple, list)):
            d_output = d_output[0]  # Take the first element if d_output is a tuple

        # Store var in id_to_variable mapping if it's not already there
        if var.unique_id not in id_to_variable:
            id_to_variable[var.unique_id] = var

        # If this is a leaf node, accumulate its derivative
        if var.is_leaf():
            id_to_variable[var.unique_id].accumulate_derivative(d_output)
        else:
            # Otherwise, propagate the gradient to the inputs using the chain rule
            for parent_var, local_grad in var.chain_rule(d_output):
                parent_id = (
                    parent_var.unique_id
                )  # Use unique_id as the key in the gradients dictionary

                # Accumulate gradients for parent_var
                if parent_id in gradients:
                    gradients[parent_id] += local_grad
                else:
                    gradients[parent_id] = local_grad

                # Add parent_var to id_to_variable mapping
                if parent_id not in id_to_variable:
                    id_to_variable[parent_id] = parent_var


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values for backward pass.

        Returns
        -------
            Tuple[Any, ...]: The values saved during the forward pass.

        """
        return self.saved_values
