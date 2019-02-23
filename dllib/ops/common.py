from .mathops import ExpOp, LnOp, MeanOp, AbsOp


def exp(op):
    return ExpOp(op)


def log(op):
    return LnOp(op)


def reduce_mean(op):
    return MeanOp(op)


def add(op1, op2):
    return op1 + op2


def sub(op1, op2):
    return op1 - op2


def mul(op1, op2):
    return op1 * op2


def mmul(op1, op2):
    return op1 @ op2


def abs(op):
    return AbsOp(op)
