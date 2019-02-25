"""
This file contains core operations, defined as classes inherited from basic classes in IOps.py
"""

from .IOps import IOperation, ITrainable, UnaryOp, BinaryOp, ComputableOp
from numpy import ndarray, outer, squeeze, sum, zeros, array
from .exceptions import InvalidShapeError
from dllib.util import element_wise_binary
import dllib.util as util


class Constant(IOperation):
    """
    This is class for constant variables, which can not be updated
    """

    def __init__(self, value: ndarray):
        super().__init__()
        self.value = value

    def forward(self):
        return self.value

    def backward(self, gradient: ndarray):
        pass

    def get_variables(self):
        return set()

    def check_shape(self):
        return self.value.shape

    def reset(self):
        pass


class Placeholder(Constant):
    """
    Something like tf.placeholder, useful in SGD
    """

    def __init__(self, shape: tuple):
        super().__init__(zeros(shape))

    def feed(self, val: ndarray):
        if val.shape != self.value.shape:
            raise InvalidShapeError("wrong shape when feed placeholder: "
                                    "placeholder shape: {0}, data shape: {1}".format(self.value.shape, val.shape))
        self.value = val


class Variable(ComputableOp, ITrainable):
    """
    Class for trainable parameters
    """

    def __init__(self, value: ndarray, name: str, trainable: bool = True):
        super().__init__()
        self.value = value
        self.name = name
        self.trainable = trainable

    def __hash__(self):
        return self.name.__hash__()

    def forward(self):
        return self.value

    def backward(self, gradient: ndarray):
        self.add_gradient(gradient)

    def get_variables(self):
        s = set()
        s.add(self)
        return s

    def check_shape(self):
        return self.value.shape

    def apply_gradient(self, gradient: ndarray, lr: float):
        self.value -= gradient * lr

    def set_value(self, val: ndarray):
        self.value = val

    def reset(self):
        super().reset()

    def compute_value(self) -> ndarray:
        pass

    def compute_gradient(self):
        pass

    def __eq__(self, other):
        return self.name == other.name


class AddOp(BinaryOp):
    """
    Class for add operations
    """

    def __init__(self, op1: IOperation, op2: IOperation):
        super().__init__(op1, op2)

    def valid_shape(self, shape1: tuple, shape2: tuple):
        """
        Pay attention to broadcast
        :param shape1: out-shape of input 1
        :param shape2: out-shape of input 2
        :return: (False, None) if not compatible, (True, out-shape) else
        """
        return element_wise_binary(shape1, shape2)

    def compute_value(self):
        return self.op1.forward() + self.op2.forward()

    def compute_gradient(self):
        shape_op1 = self.op1.forward().shape
        shape_op2 = self.op2.forward().shape
        re = util.is_broadcast(shape_op1, shape_op2)
        if re == util.NOT_BROADCAST:
            return self.grad, self.grad
        elif re == util.BROADCAST_1:
            g = sum(self.grad, axis=0)
            return self.grad, g
        elif re == util.BROADCAST_2:
            g = sum(self.grad, axis=0)
            return g, self.grad
        elif re == util.BROADCAST_1_SCALAR:
            g = sum(self.grad)
            return self.grad, array([g])
        elif re == util.BROADCAST_2_SCALAR:
            g = sum(self.grad)
            return array([g]), self.grad
        else:
            raise InvalidShapeError("wrong gradient shape: {0} and variable shape: {1}, {2}".
                                    format(self.grad.shape, shape_op1, shape_op2))


class SubOp(BinaryOp):
    """
    Class for subtract operations
    """

    def __init__(self, op1: IOperation, op2: IOperation):
        super().__init__(op1, op2)

    def valid_shape(self, shape1: tuple, shape2: tuple):
        """
        Pay attention to broadcast
        :param shape1: out-shape of input 1
        :param shape2: out-shape of input 2
        :return: (False, None) if not compatible, (True, out-shape) else
        """
        return element_wise_binary(shape1, shape2)

    def compute_value(self):
        return self.op1.forward() - self.op2.forward()

    def compute_gradient(self):
        shape_op1 = self.op1.forward().shape
        shape_op2 = self.op2.forward().shape
        re = util.is_broadcast(shape_op1, shape_op2)
        if re == util.NOT_BROADCAST:
            return self.grad, -self.grad
        elif re == util.BROADCAST_1:
            g = sum(self.grad, axis=0)
            return self.grad, -g
        elif re == util.BROADCAST_2:
            g = sum(self.grad, axis=0)
            return g, -self.grad
        elif re == util.BROADCAST_1_SCALAR:
            g = sum(self.grad)
            return self.grad, -array([g])
        elif re == util.BROADCAST_2_SCALAR:
            g = sum(self.grad)
            return array([g]), -self.grad
        else:
            raise InvalidShapeError("wrong gradient shape: {0} and variable shape: {1}, {2}".
                                    format(self.grad.shape, shape_op1, shape_op2))


class MulNumOp(UnaryOp):
    """
    Class for operations that multiply a number
    """

    def __init__(self, op: IOperation, num: float):
        super().__init__(op)
        self.num = num

    def compute_value(self) -> ndarray:
        return self.op.forward() * self.num

    def compute_gradient(self):
        return self.num * self.grad


class AddNumOp(UnaryOp):
    """
    Class for operations that add a number
    """

    def __init__(self, op: IOperation, num: float):
        super().__init__(op)
        self.num = num

    def compute_value(self) -> ndarray:
        return self.op.forward() + self.num

    def compute_gradient(self):
        return self.grad


class SubNumOp(UnaryOp):
    """
    Class for operations that sub a number
    """

    def __init__(self, op: IOperation, num: float):
        super().__init__(op)
        self.num = num

    def compute_value(self) -> ndarray:
        return self.op.forward() - self.num

    def compute_gradient(self):
        return self.grad


class RSubNumOp(UnaryOp):
    """
    Class for operations that sub a number
    """

    def __init__(self, op: IOperation, num: float):
        super().__init__(op)
        self.num = num

    def compute_value(self) -> ndarray:
        return self.num - self.op.forward()

    def compute_gradient(self):
        return -self.grad


class DivNumOp(UnaryOp):
    """
    Class for operations that multiply a number
    """

    def __init__(self, op: IOperation, num: float):
        super().__init__(op)
        self.num = num

    def compute_value(self) -> ndarray:
        return self.op.forward() / self.num

    def compute_gradient(self):
        return self.grad / self.num


class DivedNumOp(UnaryOp):
    """
    Class for operations that divided by a number
    """

    def __init__(self, op: IOperation, num: float):
        super().__init__(op)
        self.num = num

    def compute_value(self) -> ndarray:
        return self.num / self.op.forward()

    def compute_gradient(self):
        return self.grad * -self.num * (1 / self.op.forward()) ** 2


class DivOp(BinaryOp):

    def __init__(self, op1: IOperation, op2: IOperation):
        super().__init__(op1, op2)

    def valid_shape(self, shape1: tuple, shape2: tuple):
        return element_wise_binary(shape1, shape2)

    def compute_value(self):
        return self.op1.forward() / self.op2.forward()

    def compute_gradient(self):
        divisor = self.op2.forward()
        dividend = self.op1.forward()
        shape_op1 = dividend.shape
        shape_op2 = divisor.shape
        t = 1 / divisor
        re = util.is_broadcast(shape_op1, shape_op2)
        if re == util.NOT_BROADCAST:
            return self.grad * t, -self.grad * dividend * t ** 2
        elif re == util.BROADCAST_1:
            g = sum(-self.grad * dividend * t ** 2, axis=0)
            return self.grad * t, g
        elif re == util.BROADCAST_2:
            g = sum(self.grad * t, axis=0)
            return g, -self.grad * dividend * t ** 2
        elif re == util.BROADCAST_1_SCALAR:
            g = sum(-self.grad * dividend * t ** 2)
            return self.grad * t, array([g])
        elif re == util.BROADCAST_2_SCALAR:
            g = sum(self.grad * t)
            return array([g]), -self.grad * dividend * t ** 2
        else:
            raise InvalidShapeError("wrong gradient shape: {0} and variable shape: {1}, {2}".
                                    format(self.grad.shape, shape_op1, shape_op2))


class NegOp(UnaryOp):
    """
    Class for operations that negative an operation
    """

    def __init__(self, op: IOperation):
        super().__init__(op)

    def compute_value(self) -> ndarray:
        return -self.op.forward()

    def compute_gradient(self):
        return -self.grad


class MMulOp(BinaryOp):
    """
    This is vector multiply operation, in the form of
    y = Ax, where y is m*k, A is m*n, x is n*k
    """

    def __init__(self, data: IOperation, vec: IOperation):
        super().__init__(data, vec)

    def valid_shape(self, shape1: tuple, shape2: tuple) -> (bool, tuple):
        if len(shape1) == 2 and len(shape2) == 1:
            if shape1[1] == shape2[0]:
                return True, (shape1[0],)
            else:
                return False, None
        elif len(shape1) == 1 and len(shape2) == 1 and shape1[0] == shape2[0]:
            return True, (1,)
        elif len(shape1) == 2 and len(shape2) == 2:
            if shape1[1] == shape2[0]:
                return True, (shape1[0], shape2[1])
            else:
                return False, None
        else:
            return False, None

    def compute_value(self) -> ndarray:
        x1 = self.op1.forward()
        x2 = self.op2.forward()
        if len(x1.shape) == 1 and len(x2.shape) == 1:
            return array(x1.dot(x2))
        else:
            return x1.dot(x2)

    def compute_gradient(self):
        A = self.op1.forward()
        x = self.op2.forward()

        if len(A.shape) == 1:
            return squeeze(outer(self.grad, x)), self.grad * A
        elif len(self.grad.shape) == 1:
            return squeeze(outer(self.grad, x)), A.T.dot(self.grad)
        else:
            return self.grad.dot(x.T), A.T.dot(self.grad)


class MulOp(BinaryOp):
    """
    class for element-wise multiply
    """

    def __init__(self, op1: IOperation, op2: IOperation):
        super().__init__(op1, op2)

    def valid_shape(self, shape1: tuple, shape2: tuple):
        return element_wise_binary(shape1, shape2)

    def compute_value(self):
        return self.op1.forward() * self.op2.forward()

    def compute_gradient(self):
        shape_op1 = self.op1.forward().shape
        shape_op2 = self.op2.forward().shape
        re = util.is_broadcast(shape_op1, shape_op2)
        if re == util.NOT_BROADCAST:
            return self.grad * self.op2.forward(), self.grad * self.op1.forward()
        elif re == util.BROADCAST_1:
            g = sum(self.grad * self.op1.forward(), axis=0)
            return self.grad * self.op2.forward(), g
        elif re == util.BROADCAST_2:
            g = sum(self.grad * self.op2.forward(), axis=0)
            return g, self.grad * self.op1.forward()
        elif re == util.BROADCAST_1_SCALAR:
            g = sum(self.grad * self.op1.forward())
            return self.grad * self.op2.forward(), array([g])
        elif re == util.BROADCAST_2_SCALAR:
            g = sum(self.grad * self.op2.forward())
            return array([g]), self.grad * self.op1.forward()
        else:
            raise InvalidShapeError("wrong gradient shape: {0} and variable shape: {1}, {2}".
                                    format(self.grad.shape, shape_op1, shape_op2))
