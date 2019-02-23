import numpy as np
import dllib as dlib
from dllib.optimizer import GradientDescent

N = 1000
w0 = np.array([1.0, 2.0])
b0 = np.array([4.0])

X = np.random.randn(N, 2)
Y = np.dot(X, w0) + b0

data = dlib.Placeholder((N, 2))
data.feed(X)

label = dlib.Placeholder((N,))
label.feed(Y)

w = dlib.Variable(np.array([0.0, 0.0]), name="w")
b = dlib.Variable(np.array([0.0]), name="b")

pred = data@w + b
loss = dlib.reduce_mean(pred*pred)

optimizer = GradientDescent(loss, 0.01)
optimizer.train(10, verbose=False)

print("w is: ", w.forward())
print("b is: ", b.forward())
