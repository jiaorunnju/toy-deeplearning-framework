from ops.IOps import IOperation, ITrainable, UnaryOp, BinaryOp
from numpy import ndarray, outer, squeeze, sum, zeros

from ops.exceptions import InvalidShapeError
from util.help_func import merge_gradient


class Constant(IOperation):
    """
    This is class for constant variables, which can not be updated
    """

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


class Variable(IOperation, ITrainable):
    """
    Class for trainable parameters
    """

    def __init__(self, value: ndarray, name: str, trainable: bool = True):
        self.value = value
        self.name = name
        self.trainable = trainable

    def __hash__(self):
        return self.name.__hash__()

    def forward(self):
        return self.value

    def backward(self, gradient: ndarray):
        """
        pay attention to broadcast operations
        :param gradient: gardient from up-streams
        :return:
        """
        if not self.trainable:
            return {}
        else:
            if gradient.shape == self.value.shape:
                return {self.name: gradient}
            else:
                g = sum(gradient, axis=0)
                if g.shape == squeeze(self.value).shape:
                    return {self.name: sum(gradient, axis=0)}
                else:
                    raise InvalidShapeError("wrong gradient shape: {0} and variable shape: {1}".
                                            format(gradient.shape, self.value.shape))

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
        else:
            return False, None

    def compute(self):
        return self.op1.forward() + self.op2.forward()

    def backward(self, gradient: ndarray):
        return merge_gradient(self.op1.backward(gradient), self.op2.backward(gradient))


class SubOp(BinaryOp):
    """
    Class for subtraction operations
    """

    def __init__(self, op1: IOperation, op2: IOperation):
        super().__init__(op1, op2)

    def valid_shape(self, shape1: tuple, shape2: tuple):
        if shape1 == shape2:
            return True, shape1
        else:
            return False, None

    def compute(self):
        return self.op1.forward() - self.op2.forward()

    def backward(self, gradient: ndarray):
        return merge_gradient(self.op1.backward(gradient), self.op2.backward(-gradient))


class MulNumOp(UnaryOp):
    """
    Class for operations that multiply a number
    """

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

        if len(A.shape) == 1:
            return merge_gradient(self.op1.backward(squeeze(outer(gradient, x))),
                                  self.op2.backward(gradient * A))
        else:
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
