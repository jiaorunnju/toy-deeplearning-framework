from dllib.ops import IOperation
from abc import abstractmethod
from numpy import array


class Optimizer:
    """
    Class for optimizers
    """

    def __init__(self, loss: IOperation):
        self.loss = loss

    @abstractmethod
    def train(self, n_rounds: int):
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

    def train(self, n_rounds: int, verbose: bool = True, interval: int = 50):
        var = self.loss.get_variables()
        for i in range(n_rounds):
            err = self.loss.forward()
            self.loss.backward(array([1.0]))
            if verbose and i % interval == 0:
                print("[{0:5d}/{1}] loss: {2:4f}".format(i, n_rounds, err))
            for v in var:
                print(v.grad)
                v.apply_gradient(v.grad, self.lr)
            self.loss.reset()
