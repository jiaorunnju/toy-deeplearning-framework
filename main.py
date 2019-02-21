import numpy as np
from ops import *
from optimizer import GradientDescent

N = 1000
w0 = np.array([1.0, 2.0])
b0 = np.array([4.0])

X = np.random.randn(N, 2)
Y = np.dot(X, w0) + b0

data = Placeholder((N, 2))
data.feed(X)

label = Placeholder((N,))
label.feed(Y)

w = Variable(np.array([0.0, 0.0]), name="w")
b = Variable(np.array([0.8]), name="b")

pred = VMulOp(data, w)
pred = pred + b
m = pred - label
loss = VMulOp(m, m)
loss = loss*(1/N)

optimizer = GradientDescent(loss, 0.01)
optimizer.train(500)

print("w is: ", w.forward())
print("b is: ", b.forward())
