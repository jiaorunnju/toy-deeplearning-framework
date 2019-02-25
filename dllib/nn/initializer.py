from numpy.random import randn


def gaussian_initializer(indim, outdim):
    if outdim == 1:
        return randn(indim)
    else:
        return randn(indim, outdim)
