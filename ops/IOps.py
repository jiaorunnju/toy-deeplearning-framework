from abc import ABCMeta, abstractmethod, ABC
from numpy import ndarray

from ops.exceptions import InvalidShapeError
import traceback
import sys


class IOperation:
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self) -> ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, gradient: ndarray) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_variables(self) -> set:
        raise NotImplementedError

    @abstractmethod
    def check_shape(self) -> tuple:
        raise NotImplementedError

    def reset(self):
        pass


class ITrainable:
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply_gradient(self, gradient: ndarray, lr: float):
        raise NotImplementedError


class ComputableOp(IOperation):

    def __init__(self):
        self.computed: bool = False
        self.data: ndarray = None

    @abstractmethod
    def compute(self) -> ndarray:
        raise NotImplementedError

    def reset(self):
        self.computed = False

    def forward(self):
        if not self.computed:
            self.data = self.compute()
            self.computed = True
        return self.data


class UnaryOp(ComputableOp, ABC):

    def __init__(self, op: IOperation):
        super().__init__()
        self.op = op

    def get_variables(self):
        return self.get_variables()

    def check_shape(self):
        try:
            s = self.op.check_shape()
            return s
        except InvalidShapeError as err:
            print(err.message)
            traceback.print_exc()
            sys.exit(sys.exit(1))


class BinaryOp(ComputableOp, ABC):

    def __init__(self, op1: IOperation, op2: IOperation):
        super().__init__()
        self.op1 = op1
        self.op2 = op2

    def get_variables(self):
        return self.op1.get_variables().union(self.op2.get_variables())

    def valid_shape(self, shape1: tuple, shape2: tuple) -> (bool, tuple):
        raise NotImplementedError

    def check_shape(self):
        try:
            s1 = self.op1.check_shape()
            s2 = self.op2.check_shape()
            valid, outshape = self.valid_shape(s1, s2)
            if not valid:
                raise InvalidShapeError("wrong shape in {0} and {1}".format(s1, s2))
            return outshape
        except InvalidShapeError as err:
            print(err.message)
            traceback.print_exc()
            sys.exit(sys.exit(1))
