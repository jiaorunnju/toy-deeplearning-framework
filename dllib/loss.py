"""
This file contains different losses
"""

from .ops import *


def mse_loss(pred, y):
    return reduce_mean((pred - y) * (pred - y))


def absolute_loss(pred, y):
    return reduce_mean(abs(pred - y))
