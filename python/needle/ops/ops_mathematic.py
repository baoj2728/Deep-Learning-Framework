"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a ** b
        
    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad * (rhs * lhs ** (rhs - 1)), out_grad * (lhs ** rhs * log(lhs))

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        lhs, = node.inputs
        return (out_grad * (self.scalar * lhs ** (self.scalar - 1)), ) 


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (out_grad * 1 / rhs, out_grad * (-lhs) / (rhs ** 2))


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        lhs, = node.inputs
        return (out_grad / self.scalar,)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        else:
            return array_api.swapaxes(a, -2, -1)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.reshape(self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        output_shape = self.shape

        axes = []
        offset = len(output_shape) - len(input_shape)
        axes.extend(range(offset))

        for i, dim in enumerate(input_shape):
            if dim == 1 and output_shape[i + offset] != 1:
                axes.append(i + offset)

        grad = summation(out_grad, tuple(axes))

        return reshape(grad, input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape

        if self.axes:
            reshape_shape = list(input_shape)
            for axis in self.axes:
                reshape_shape[axis] = 1
            reshape_shape = tuple(reshape_shape)

            return broadcast_to(
                reshape(out_grad, reshape_shape),
                input_shape
            )
        else:
            return broadcast_to(out_grad, input_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs

        grad_lhs = out_grad @ transpose(rhs)
        grad_rhs = transpose(lhs) @ out_grad

        def sum_to_shape(grad, shape):
            axes = []

            # Extra leading dimensions introduced by broadcasting
            extra = len(grad.shape) - len(shape)
            axes += list(range(extra))

            # Dimensions of size 1 that were broadcast
            for i in range(len(shape)):
                if shape[i] == 1 and grad.shape[i + extra] != 1:
                    axes.append(i + extra)

            if axes:
                grad = summation(grad, axes=tuple(axes))
                grad = reshape(grad, shape)

            return grad

        grad_lhs = sum_to_shape(grad_lhs, lhs.shape)
        grad_rhs = sum_to_shape(grad_rhs, rhs.shape)

        return grad_lhs, grad_rhs


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return (-out_grad, )


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        lhs, = node.inputs
        return (out_grad / lhs, )


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        lhs, = node.inputs
        return (out_grad * exp(lhs), )


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(0, a)

    def gradient(self, out_grad, node):
        mask = Tensor(
            (node.inputs[0].realize_cached_data() > 0).astype(
                node.inputs[0].realize_cached_data().dtype
            )
        )
        return out_grad * mask


def relu(a):
    return ReLU()(a)

