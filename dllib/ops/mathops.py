"""
This file contains classes for common math operations
"""

from numpy import ndarray
from numpy import exp, log, mean, ones_like, abs, sign, power

from .IOps import UnaryOp, IOperation


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
        return self.grad * 1/self.op.forward()


class MeanOp(UnaryOp):

    def __init__(self, op: IOperation):
        super().__init__(op)

    def compute_value(self):
        return mean(self.op.forward())

    def compute_gradient(self):
        t = self.op.forward()
        return self.grad * ones_like(t) / t.size


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
