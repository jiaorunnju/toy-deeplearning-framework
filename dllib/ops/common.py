import dllib.ops.mathops as mathops


def exp(op):
    return mathops.ExpOp(op)


def log(op):
    return mathops.LnOp(op)


def reduce_mean(op):
    return mathops.MeanOp(op)


def add(op1, op2):
    return op1 + op2


def sub(op1, op2):
    return op1 - op2


def mul(op1, op2):
    return op1 * op2


def mmul(op1, op2):
    return op1 @ op2


def abs(op):
    return mathops.AbsOp(op)
