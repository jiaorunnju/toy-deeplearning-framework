import numpy as np
import dllib
from dllib.optimizer import GradientDescent
from dllib.loss import logistic_loss_with_logits
import matplotlib.pyplot as plt
from dllib.nn.layer import dense
from dllib.nn.activations import sigmoid

N = 100
DIM = 2
x1 = np.random.randn(N, DIM)
x2 = np.random.randn(N, DIM)
y1 = np.ones(N)
y2 = -np.ones(N)
x1 += np.array([1, 1])
x2 += np.array([-1, -1])

X = np.concatenate((x1, x2), axis=0)
Y = np.concatenate((y1, y2))

data = dllib.Placeholder((2 * N, 2))
data.feed(X)
label = dllib.Placeholder((2 * N,))
label.feed(Y)

t = dense(data, 2, 3, "dense1", sigmoid)
t = dense(t, 3, 1, "dense2")

t.check_shape()

loss = logistic_loss_with_logits(t, label)

optimizer = GradientDescent(loss, 0.01)
optimizer.optimize(1000)

pred = np.sign(t.forward())
acc = pred[((pred - Y) == 0)].size/(2*N)
print("accuracy is: {0}".format(acc))

plt.plot(X[:N, 0], X[:N, 1], 'r*')
plt.plot(X[N:2*N, 0], X[N:2*N, 1], 'g*')
plt.show()