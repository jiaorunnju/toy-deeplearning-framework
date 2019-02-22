from dllib.ops import UnaryOp, IOperation
from numpy import ndarray, abs, sum, sign


class MSE(UnaryOp):
    """
    Class for mse loss
    """

    def __init__(self, pred: IOperation, label: IOperation):
        super().__init__(pred)
        self.label = label

    def compute(self):
        t = self.op.forward() - self.label.forward()
        n = len(t)
        return t.dot(t)/n

    def backward(self, gradient: ndarray):
        t = self.op.forward() - self.label.forward()
        n = len(t)
        return self.op.backward(2 / n * gradient * t)


class AbsoluteLoss(UnaryOp):
    """
    Class for mse loss
    """

    def __init__(self, pred: IOperation, label: IOperation):
        super().__init__(pred)
        self.label = label

    def compute(self):
        t = self.op.forward() - self.label.forward()
        n = len(t)
        return sum(abs(t))/n

    def backward(self, gradient: ndarray):
        t = self.op.forward() - self.label.forward()
        n = len(t)
        return self.op.backward(2 / n * gradient * sign(t))
