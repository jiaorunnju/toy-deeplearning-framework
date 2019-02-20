from numpy.core.multiarray import ndarray

from ops.IOps import UnaryOp


class sigmoid(UnaryOp):
    def compute(self) -> ndarray:
        pass

    def backward(self, gradient: ndarray) -> dict:
        pass
