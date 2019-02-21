from abc import ABCMeta, abstractmethod, ABC
from numpy import ndarray

from ops.exceptions import InvalidShapeError
import traceback
import sys


class IOperation:
    """
    This is interface for operations. Operations are used to define computation graphs
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self) -> ndarray:
        """
        forward the dataflow in graph
        :return: result of current node in graph
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, gradient: ndarray) -> dict:
        """
        backward the gradient and get the gradient of variables with respect to loss
        :param gradient: gradient from up-stream
        :return: a dict of variables' names and their gradient, e.g. {"w": [1.0,1.0]}
        """
        raise NotImplementedError

    @abstractmethod
    def get_variables(self) -> set:
        """
        get trainable variables of sub-graph with respect to current node
        :return: set of trainable variables
        """
        raise NotImplementedError

    @abstractmethod
    def check_shape(self) -> tuple:
        """
        check whether the shape of nodes in current sub-graph are compatible
        :return: output-shape of current node if compatible, or raise error if not
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        reset the graph, thus need to re-compute the value of different nodes when forward
        :return:
        """
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, IOperation):
            from ops.operations import AddOp
            return AddOp(self, other)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, IOperation):
            from ops.operations import SubOp
            return SubOp(self, other)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            from ops.operations import MulNumOp
            return MulNumOp(self, other)
        else:
            raise NotImplementedError


class ITrainable:
    """
    This is interface for trainable variables with gradient
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply_gradient(self, gradient: ndarray, lr: float):
        """
        apply the gradient and update the variable
        :param gradient: gradient of variables
        :param lr: learning rate
        :return:
        """
        raise NotImplementedError


class ComputableOp(IOperation):
    """
    This is the class represent computable operations, such as plus, minus, etc.
    """

    def __init__(self):
        self.computed: bool = False
        self.data: ndarray = None

    @abstractmethod
    def compute(self) -> ndarray:
        """
        compute output of current node
        :return: output of current node
        """
        raise NotImplementedError

    def forward(self):
        """
        forward with cached result
        :return: cached result
        """
        if not self.computed:
            self.data = self.compute()
            self.computed = True
        return self.data


class UnaryOp(ComputableOp, ABC):
    """
    This is class for operations with only one inputs, such as activation in DNN
    """

    def __init__(self, op: IOperation):
        super().__init__()
        self.op = op

    def get_variables(self):
        return self.op.get_variables()

    def reset(self):
        self.computed = False
        # logging.warn("reset called")
        self.op.reset()

    def check_shape(self):
        try:
            s = self.op.check_shape()
            return s
        except InvalidShapeError as err:
            print(err.message)
            traceback.print_exc()
            sys.exit(sys.exit(1))


class BinaryOp(ComputableOp, ABC):
    """
    This is class for operations with two inputs, such as plus.
    """

    def __init__(self, op1: IOperation, op2: IOperation):
        super().__init__()
        self.op1 = op1
        self.op2 = op2

    def get_variables(self):
        return self.op1.get_variables().union(self.op2.get_variables())

    def valid_shape(self, shape1: tuple, shape2: tuple) -> (bool, tuple):
        """
        Check whether the shape of two inputs are compatible
        :param shape1: out-shape of input 1
        :param shape2: out-shape of input 2
        :return: (False, None) if not compatible, (True, out-shape) else
        """
        raise NotImplementedError

    def reset(self):
        self.computed = False
        self.op1.reset()
        self.op2.reset()

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
