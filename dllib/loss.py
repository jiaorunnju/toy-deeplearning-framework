from dllib.ops import UnaryOp, IOperation
from numpy import ndarray


class MSE(UnaryOp):
    """
    Class for mse loss
    """

    def __init__(self, pred: IOperation, label: IOperation):
        super().__init__(pred)
        self.label = label

    def compute(self):
        t = self.op.forward()-self.label.forward()
        n = len(t)
        return 1 / n * t.dot(t)

    def backward(self, gradient: ndarray):
        t = self.op.forward() - self.label.forward()
        n = len(t)
        return self.op.backward(2 / n * gradient * t)
