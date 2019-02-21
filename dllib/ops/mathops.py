from numpy.core.multiarray import ndarray
from numpy import exp, log

from dllib.ops import UnaryOp, IOperation


class ExpOp(UnaryOp):

    def __init__(self, op: IOperation):
        super().__init__(op)

    def compute(self) -> ndarray:
        return exp(self.op.forward())

    def backward(self, gradient: ndarray) -> dict:
        return self.op.backward(exp(gradient))


class LnOp(UnaryOp):

    def __init__(self, op: IOperation):
        super().__init__(op)

    def compute(self) -> ndarray:
        return log(self.op.forward())

    def backward(self, gradient: ndarray) -> dict:
        return self.op.backward(1.0 / gradient)
