"""
This file contains useful math operations, use functions as wrappers
"""

from .mathops import ExpOp, LnOp, MeanOp, AbsOp, PowOp, MaxOp, MinOp, MaxNumOp, MinNumOp, SumOp


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


def div(op1, op2):
    return op1 / op2


def pow(op1, op2):
    return PowOp(op1, op2)


def max(op1, op2):
    if isinstance(op1, (float, int)):
        return MaxNumOp(op2, op1)
    elif isinstance(op2, (float, int)):
        return MaxNumOp(op1, op2)
    else:
        return MaxOp(op1, op2)


def min(op1, op2):
    if isinstance(op1, (float, int)):
        return MinNumOp(op2, op1)
    elif isinstance(op2, (float, int)):
        return MinNumOp(op1, op2)
    else:
        return MinOp(op1, op2)


def sum(op):
    return SumOp(op)