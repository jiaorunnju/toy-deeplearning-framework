"""
This file contains core classes for Operations, and the basic data flow operations when computing
values and gradients
"""

from abc import ABCMeta, abstractmethod, ABC
from numpy import ndarray
from .exceptions import InvalidShapeError
import traceback
import sys


class IOperation:
    """
    This is interface for operations. Operations are used to define computation graphs
    """

    def __init__(self):
        self.num_child = 0

    def add_child_count(self):
        self.num_child += 1

    @abstractmethod
    def forward(self) -> ndarray:
        """
        forward the dataflow in graph
        :return: result of current node in graph
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, gradient: ndarray):
        """
        backward the gradient and get the gradient of variables with respect to loss
        :param gradient: gradient from up-stream
        :return:
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
            from .operations import AddOp
            return AddOp(self, other)
        elif isinstance(other, (float, int)):
            from .operations import AddNumOp
            return AddNumOp(self, other)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, IOperation):
            from .operations import SubOp
            return SubOp(self, other)
        elif isinstance(other, (float, int)):
            from .operations import SubNumOp
            return SubNumOp(self, other)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            from .operations import MulNumOp
            return MulNumOp(self, other)
        elif isinstance(other, IOperation):
            from .operations import MulOp
            return MulOp(self, other)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            from .operations import MulNumOp
            return MulNumOp(self, other)
        elif isinstance(other, IOperation):
            from .operations import MulOp
            return MulOp(self, other)
        else:
            raise NotImplementedError

    def __neg__(self):
        from .operations import NegOp
        return NegOp(self)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            from .operations import DivNumOp
            return DivNumOp(self, other)
        else:
            raise NotImplementedError

    def __matmul__(self, other):
        if isinstance(other, IOperation):
            from .operations import MMulOp
            return MMulOp(self, other)
        else:
            raise NotImplementedError

    def __pow__(self, power, modulo=None):
        if isinstance(power, int):
            from .mathops import PowOp
            return PowOp(self, power)
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
        super().__init__()
        self.value_computed: bool = False
        self.value: ndarray = None
        self.gradient_computed: bool = False
        self.grad: ndarray = None
        self.grad_count = 0

    @abstractmethod
    def compute_value(self) -> ndarray:
        """
        compute output of current node
        :return: output of current node
        """
        raise NotImplementedError

    def reset(self):
        self.gradient_computed = False
        self.value_computed = False
        self.grad_count = 0

    def forward(self):
        """
        forward with cached result
        :return: cached result
        """
        if not self.value_computed:
            self.value = self.compute_value()
            self.value_computed = True
        return self.value

    def add_gradient(self, gradient: ndarray):
        """
        Add the gradient to current node
        :param gradient: gradient from up-stream
        :return:
        """
        self.grad_count += 1
        if not self.gradient_computed:
            self.grad = gradient
            self.gradient_computed = True
        else:
            self.grad = self.grad + gradient

    @abstractmethod
    def compute_gradient(self):
        """
        Compute the gradient to nodes in the front using current node's gradient
        :return: gradient with respect to each node before
        """
        raise NotImplementedError


class UnaryOp(ComputableOp, ABC):
    """
    This is class for operations with only one inputs, such as activation in DNN
    """

    def __init__(self, op: IOperation):
        super().__init__()
        op.add_child_count()
        self.op = op

    def get_variables(self):
        return self.op.get_variables()

    def reset(self):
        super().reset()
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

    def backward(self, gradient: ndarray):
        """
        Propagate the gradient only if all nodes that use this node have added their gradient
        to current node, to get a polynomial time computation
        :param gradient: gradient from up-stream
        :return:
        """
        self.add_gradient(gradient)
        if self.grad_count >= self.num_child:
            self.op.backward(self.compute_gradient())


class BinaryOp(ComputableOp, ABC):
    """
    This is class for operations with two inputs, such as plus.
    """

    def __init__(self, op1: IOperation, op2: IOperation):
        super().__init__()
        op1.add_child_count()
        op2.add_child_count()
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
        super().reset()
        self.op1.reset()
        self.op2.reset()

    def backward(self, gradient: ndarray):
        """
        Propagate the gradient only if all nodes that use this node have added their gradient
        to current node, to get a polynomial time computation
        :param gradient: gradient from up-stream
        :return:
        """
        self.add_gradient(gradient)
        if self.grad_count >= self.num_child:
            grad1, grad2 = self.compute_gradient()
            self.op1.backward(grad1)
            self.op2.backward(grad2)

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
