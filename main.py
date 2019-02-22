import numpy as np
from dllib.ops import *
from dllib.optimizer import GradientDescent
from dllib.loss import AbsoluteLoss, MSE

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
b = Variable(np.array([0.0]), name="b")

pred = VMulOp(data, w)
pred = pred + b
loss = AbsoluteLoss(pred, label)

optimizer = GradientDescent(loss, 0.01)
optimizer.train(500)

print("w is: ", w.forward())
print("b is: ", b.forward())
