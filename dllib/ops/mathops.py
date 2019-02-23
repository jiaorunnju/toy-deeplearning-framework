from numpy.core.multiarray import ndarray
from numpy import exp, log, mean, ones_like, abs, sign

from dllib.ops import UnaryOp, IOperation


class ExpOp(UnaryOp):

    def __init__(self, op: IOperation):
        super().__init__(op)

    def compute_value(self) -> ndarray:
        return exp(self.op.forward())

    def compute_gradient(self):
        return exp(self.grad)


class LnOp(UnaryOp):

    def __init__(self, op: IOperation):
        super().__init__(op)

    def compute_value(self) -> ndarray:
        return log(self.op.forward())

    def compute_gradient(self):
        return 1.0 / self.grad


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
