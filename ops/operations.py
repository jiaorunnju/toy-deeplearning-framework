from ops.IOps import IOperation, ITrainable, UnaryOp, BinaryOp
from numpy import ndarray, outer, squeeze

from util.help_func import merge_gradient


class Constant(IOperation):

    def __init__(self, value: ndarray):
        self.value = value

    def forward(self):
        return self.value

    def backward(self, gradient: ndarray):
        return {}

    def get_variables(self):
        return set()

    def check_shape(self):
        return self.value.shape

    def set_value(self, val: ndarray):
        self.value = val


class Variable(IOperation, ITrainable):

    def __init__(self, value: ndarray, name: str, trainable: bool):
        self.value = value
        self.name = name
        self.trainable = trainable

    def forward(self):
        return self.value

    def backward(self, gradient: ndarray):
        if not self.trainable:
            return {}
        else:
            return {self.name: gradient}

    def get_variables(self):
        s = set()
        s.add(self)
        return s

    def check_shape(self):
        return self.value.shape

    def apply_gradient(self, gradient: ndarray, lr: float):
        self.value -= gradient * lr

    def __eq__(self, other):
        return self.name == other.name


class AddOp(BinaryOp):

    def __init__(self, op1: IOperation, op2: IOperation):
        super().__init__(op1, op2)

    def valid_shape(self, shape1: tuple, shape2: tuple):
        if shape1 == shape2:
            return True, shape1
        else:
            return False, None

    def compute(self):
        return self.op1.forward() + self.op2.forward()

    def backward(self, gradient: ndarray):
        return merge_gradient(self.op1.backward(gradient), self.op2.backward(gradient))


class MulNumOp(UnaryOp):

    def __init__(self, op: IOperation, num: float):
        super().__init__(op)
        self.num = num

    def compute(self) -> ndarray:
        return self.op.forward() * self.num

    def backward(self, gradient: ndarray):
        return self.op.backward(gradient * self.num)


class VMulOp(BinaryOp):
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

    def compute(self) -> ndarray:
        return self.op1.forward().dot(self.op2.forward())

    def backward(self, gradient: ndarray) -> dict:
        A = self.op1.forward()
        x = self.op2.forward()
        return merge_gradient(self.op1.backward(squeeze(outer(gradient, x))),
                              self.op2.backward(A.T.dot(gradient)))


class MMulOp(BinaryOp):
    """
    This is matrix multiply operation, in the form of
    Y = AX, where y is m*n, A is m*k, X is k*n
    """

    def __init__(self, data: IOperation, mat: IOperation):
        super().__init__(data, mat)

    def valid_shape(self, shape1: tuple, shape2: tuple) -> (bool, tuple):
        if len(shape1) != len(shape2) + 1:
            return False, None
        if len(shape1) == 3:
            if shape1[2] == shape2[0]:
                return True, (shape1[0], shape1[1], shape2[1])
            else:
                return False, None
        else:
            return False, None

    def compute(self) -> ndarray:
        return self.op1.forward().dot(self.op2.forward())

    def backward(self, gradient: ndarray) -> dict:
        raise NotImplementedError
