"""
This file contains core operations, defined as classes inherited from basic classes in IOps.py
"""

from .IOps import IOperation, ITrainable, UnaryOp, BinaryOp, ComputableOp
from numpy import ndarray, outer, squeeze, sum, zeros
from .exceptions import InvalidShapeError


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
        if shape1 == shape2:
            return True, shape1
        elif len(shape1) == len(shape2) + 1 and shape1[1:] == shape2:
            return True, shape1
        elif len(shape2) == len(shape1) + 1 and shape2[2:] == shape1:
            return True, shape2
        elif len(shape1) == 1 and len(shape2) == 1 and shape2[0] == 1:
            return True, shape1
        elif len(shape1) == 1 and len(shape2) == 1 and shape1[0] == 1:
            return True, shape2
        else:
            return False, None

    def compute_value(self):
        return self.op1.forward() + self.op2.forward()

    def compute_gradient(self):
        shape_op1 = self.op1.forward().shape
        shape_op2 = self.op2.forward().shape
        if shape_op1 == shape_op2:
            return self.grad, self.grad
        elif len(shape_op1) == len(shape_op2) + 1 and shape_op1[1:] == shape_op2:
            g = sum(self.grad, axis=0)
            return self.grad, g
        elif len(shape_op2) == len(shape_op1) + 1 and shape_op2[1:] == shape_op1:
            g = sum(self.grad, axis=0)
            return g, self.grad
        elif len(shape_op1) == 1 and len(shape_op2) == 1 and shape_op2[0] == 1:
            g = sum(self.grad, axis=0)
            return self.grad, g
        elif len(shape_op1) == 1 and len(shape_op2) == 1 and shape_op1[0] == 1:
            g = sum(self.grad, axis=0)
            return g, self.grad
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
        if shape1 == shape2:
            return True, shape1
        elif len(shape1) == len(shape2) + 1 and shape1[1:] == shape2:
            return True, shape1
        elif len(shape2) == len(shape1) + 1 and shape2[2:] == shape1:
            return True, shape2
        elif len(shape1) == 1 and len(shape2) == 1 and shape2[0] == 1:
            return True, shape1
        elif len(shape1) == 1 and len(shape2) == 1 and shape1[0] == 1:
            return True, shape2
        else:
            return False, None

    def compute_value(self):
        return self.op1.forward() - self.op2.forward()

    def compute_gradient(self):
        shape_op1 = self.op1.forward().shape
        shape_op2 = self.op2.forward().shape
        if shape_op1 == shape_op2:
            return self.grad, -self.grad
        elif len(shape_op1) == len(shape_op2) + 1 and shape_op1[1:] == shape_op2:
            g = sum(self.grad, axis=0)
            return self.grad, -g
        elif len(shape_op2) == len(shape_op1) + 1 and shape_op2[1:] == shape_op1:
            g = sum(self.grad, axis=0)
            return g, -self.grad
        elif len(shape_op1) == 1 and len(shape_op2) == 1 and shape_op2[0] == 1:
            g = sum(self.grad, axis=0)
            return self.grad, -g
        elif len(shape_op1) == 1 and len(shape_op2) == 1 and shape_op1[0] == 1:
            g = sum(self.grad, axis=0)
            return g, -self.grad
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
    y = Ax, where y is m*1, A is m*n, x is n*1
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
        else:
            return False, None

    def compute_value(self) -> ndarray:
        return self.op1.forward().dot(self.op2.forward())

    def compute_gradient(self):
        A = self.op1.forward()
        x = self.op2.forward()

        if len(A.shape) == 1:
            return squeeze(outer(self.grad, x)), self.grad * A
        else:
            return squeeze(outer(self.grad, x)), A.T.dot(self.grad)


class MulOp(BinaryOp):
    """
    class for element-wise multiply
    """

    def __init__(self, op1: IOperation, op2: IOperation):
        super().__init__(op1, op2)

    def valid_shape(self, shape1: tuple, shape2: tuple):
        if shape1 == shape2:
            return True, shape1
        else:
            return False, None

    def compute_value(self):
        return self.op1.forward() * self.op2.forward()

    def compute_gradient(self):
        return self.grad * self.op2.forward(), self.grad * self.op1.forward()
