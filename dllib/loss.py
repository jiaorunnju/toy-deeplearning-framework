"""
This file contains different losses
"""

from .ops import *
from .ops import max


def mse_loss(pred, y):
    return reduce_mean((pred - y) * (pred - y))


def absolute_loss(pred, y):
    return reduce_mean(abs(pred - y))


def logistic_loss_with_logits(logits, y):
    t = logits * y
    return -sum(log(max(1/(1+exp(-t)), 1e-7)))


