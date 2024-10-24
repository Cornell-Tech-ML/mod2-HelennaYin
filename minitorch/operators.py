"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiplies two numbers x and y"""
    return x * y


def id(x: float) -> float:
    """Returns the input number x"""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers x and y"""
    return x + y


def neg(x: float) -> float:
    """Negates a number x"""
    return -x


def lt(x: float, y: float) -> bool:
    """Compares two numbers x and y and returns True if x is less than y"""
    return x < y


def eq(x: float, y: float) -> bool:
    """Compares two numbers x and y and returns True if x is equal to y"""
    return x == y


def max(x: float, y: float) -> float:
    """Returns the maximum of two numbers x and y"""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers x and y are close"""
    return abs(x - y) < 0.01


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function of x"""
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x))


def relu(x: float) -> float:
    """Calculate the rectified linear unit (ReLU) function of x"""
    return max(0, x)


def log(x: float) -> float:
    """Calculate the natural logarithm of x"""
    return math.log(x)


def exp(x: float) -> float:
    """Calculate the exponential of x"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculate the reciprocal of x"""
    return 1 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log x times a second arg y"""
    return 1 / x * y


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of relu x times a second arg y"""
    if x < 0:
        return 0
    else:
        return y


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal x times a second arg y"""
    return -1 / (x**2) * y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable, x: Iterable[float]) -> Iterable[float]:
    """Apply a function fn to each element in an iterable x"""
    return [fn(i) for i in x]


def zipWith(fn: Callable, a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    """Combines elements from two iterables a and b using a given function fn"""
    return [fn(x, y) for x, y in zip(a, b)]


def reduce(fn: Callable, a: Iterable[float]) -> float:
    """Reduces an iterable a to a single value using a given function"""
    a = iter(a)

    temp = next(a)
    for i in a:
        temp = fn(temp, i)
    return temp


def negList(a: list) -> Iterable:
    """Negate all elements in a list"""
    return map(neg, a)


def addLists(a: list, b: list) -> Iterable:
    """Add corresponding elements from two lists"""
    return zipWith(add, a, b)


def sum(a: list) -> float:
    """Sum all elements in a list"""
    if not a:
        return 0
    else:
        return reduce(add, a)


def prod(a: list) -> float:
    """Calculate the product of all elements in a list"""
    if not a:
        return 0
    else:
        return reduce(mul, a)
