"""
This file contains layers for neural networks
"""
from ..ops import Variable
from .initializer import gaussian_initializer
from numpy import array


def dense(op, input_dim, output_dim, name, activation=None):
    _w = Variable(gaussian_initializer(input_dim, output_dim), name + "/w")
    _b = Variable(array([0.0]), name+"/b")
    if activation is None:
        return op @ _w + _b
    else:
        return activation(op @ _w + _b)
