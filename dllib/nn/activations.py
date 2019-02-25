"""
This file contains different activation functions
"""
from dllib.ops.common import max, exp


def relu(x):
    return max(0, x)


def sigmoid(x):
    return 1 / (1 + exp(-x))
