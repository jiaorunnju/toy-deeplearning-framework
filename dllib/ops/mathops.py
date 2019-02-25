"""
This file contains classes for common math operations
"""
import sys
import traceback

from numpy import ndarray
from numpy import exp, log, mean, ones_like, abs, sign, power, maximum, zeros, minimum, sum

from dllib.ops.exceptions import InvalidShapeError
from .IOps import UnaryOp, IOperation, BinaryOp
from dllib.util import element_wise_binary


class ExpOp(UnaryOp):

    def __init__(self, op: IOperation):
        super().__init__(op)

    def compute_value(self) -> ndarray:
        return exp(self.op.forward())

    def compute_gradient(self):
        return self.grad * exp(self.op.forward())


class LnOp(UnaryOp):

    def __init__(self, op: IOperation):
        super().__init__(op)

    def compute_value(self) -> ndarray:
        return log(self.op.forward())

    def compute_gradient(self):
        return self.grad * 1 / self.op.forward()


class MeanOp(UnaryOp):

    def __init__(self, op: IOperation):
        super().__init__(op)

    def compute_value(self):
        return mean(self.op.forward())

    def compute_gradient(self):
        t = self.op.forward()
        return self.grad * ones_like(t) / t.size

    def check_shape(self):
        try:
            self.op.check_shape()
            return (1,)
        except InvalidShapeError as err:
            print(err.message)
            traceback.print_exc()
            sys.exit(sys.exit(1))


class SumOp(UnaryOp):
    def __init__(self, op: IOperation):
        super().__init__(op)

    def compute_value(self):
        return sum(self.op.forward())

    def compute_gradient(self):
        t = self.op.forward()
        return self.grad * ones_like(t)

    def check_shape(self):
        try:
            self.op.check_shape()
            return (1,)
        except InvalidShapeError as err:
            print(err.message)
            traceback.print_exc()
            sys.exit(sys.exit(1))


class AbsOp(UnaryOp):

    def __init__(self, op: IOperation):
        super().__init__(op)

    def compute_value(self):
        return abs(self.op.forward())

    def compute_gradient(self):
        t = self.op.forward()
        return self.grad * sign(t)


class PowOp(UnaryOp):

    def __init__(self, op: IOperation, num: int):
        super().__init__(op)
        self.num = num

    def compute_value(self):
        return power(self.op.forward(), self.num)

    def compute_gradient(self):
        return self.grad * self.num * power(self.op.forward(), self.num - 1)


class MaxOp(BinaryOp):

    def __init__(self, op1: IOperation, op2: IOperation):
        super().__init__(op1, op2)

    def valid_shape(self, shape1: tuple, shape2: tuple) -> (bool, tuple):
        if shape1 == shape2:
            return True, shape1
        else:
            return False, None

    def compute_value(self) -> ndarray:
        return maximum(self.op1.forward(), self.op2.forward())

    def compute_gradient(self):
        t = self.value
        v1 = self.op1.forward()
        v2 = self.op2.forward()
        s = t.shape
        g1 = zeros(s)
        g2 = zeros(s)
        g1[t == v1] = 1
        g2[t == v2] = 1
        return self.grad * g1, self.grad * g2


class MaxNumOp(UnaryOp):

    def __init__(self, op: IOperation, num: float):
        super().__init__(op)
        self.num = num

    def compute_value(self) -> ndarray:
        return maximum(self.op.forward(), self.num)

    def compute_gradient(self):
        v = self.value
        v1 = self.op.forward()
        g = zeros(v.shape)
        g[v == v1] = 1
        return self.grad * g


class MinNumOp(UnaryOp):

    def __init__(self, op: IOperation, num: float):
        super().__init__(op)
        self.num = num

    def compute_value(self) -> ndarray:
        return minimum(self.op.forward(), self.num)

    def compute_gradient(self):
        v = self.value
        v1 = self.op.forward()
        g = zeros(v.shape)
        g[v == v1] = 1
        return self.grad * g


class MinOp(BinaryOp):

    def __init__(self, op1: IOperation, op2: IOperation):
        super().__init__(op1, op2)

    def valid_shape(self, shape1: tuple, shape2: tuple) -> (bool, tuple):
        if shape1 == shape2:
            return True, shape1
        else:
            return False, None

    def compute_value(self) -> ndarray:
        return minimum(self.op1.forward(), self.op2.forward())

    def compute_gradient(self):
        t = self.value
        v1 = self.op1.forward()
        v2 = self.op2.forward()
        s = t.shape
        g1 = zeros(s)
        g2 = zeros(s)
        g1[t == v1] = 1
        g2[t == v2] = 1
        return self.grad * g1, self.grad * g2
