"""
This file contains classes for optimization methods
"""

from .ops import IOperation
from abc import abstractmethod
from numpy import array, squeeze


class Optimizer:
    """
    Class for optimizers
    """

    def __init__(self, loss: IOperation):
        self.loss = loss

    @abstractmethod
    def optimize(self, n_rounds: int):
        """
        perform optimization on a loss
        :param n_rounds: number of rounds
        :return:
        """
        raise NotImplementedError


class GradientDescent(Optimizer):
    """
    Class for gradient descent optimization methods
    """

    def __init__(self, loss: IOperation, lr: float):
        super().__init__(loss)
        self.lr = lr

    def optimize(self, n_rounds: int, verbose: bool = True, interval: int = 50):
        var = self.loss.get_variables()
        num_len = len(str(n_rounds))
        for i in range(n_rounds):
            err = self.loss.forward()
            self.loss.backward(array([1.0]))
            if verbose and i % interval == 0:
                print("[{0:{1}d}/{2}] loss: {3:4f}".format(i, num_len, n_rounds, squeeze(err)))
            for v in var:
                v.apply_gradient(v.grad, self.lr)

            self.loss.reset()
