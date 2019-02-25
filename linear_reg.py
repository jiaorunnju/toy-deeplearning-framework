import numpy as np
import dllib
from dllib.optimizer import GradientDescent

N = 1000
w0 = np.array([1.0, 2.0])
b0 = np.array([4.0])

X = np.random.randn(N, 2)
Y = np.dot(X, w0) + b0

data = dllib.Placeholder((N, 2))
data.feed(X)

label = dllib.Placeholder((N,))
label.feed(Y)

w = dllib.Variable(np.array([0.0, 0.0]), name="w")
b = dllib.Variable(np.array([0.0]), name="b")

pred = b + data @ w
loss = dllib.reduce_mean((pred-label)*(pred-label))
loss.check_shape()

optimizer = GradientDescent(loss, 0.01)
optimizer.optimize(500, verbose=True)

print("w is: ", w.forward())
print("b is: ", b.forward())