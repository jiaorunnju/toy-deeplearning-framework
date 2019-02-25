# A toy deep-learning framework in python

This is a toy deep-learning framework in python, built on top 
of numpy. It has several features:

- compute gradient automatically
- written in pure python

Here are some related papers or blogs on automatic differentiation:

- [Automatic differentiation in machine learning](http://www.jmlr.org/papers/volume18/17-468/17-468.pdf)
- [Automatic differentiation](http://www.columbia.edu/~ahd2125/post/2015/12/5/)

The main idea is to abstract different kinds of operations and
define **backward** methods for them in order to make gradient
flow through the graph.

Below are some functions or methods that this framework are
expected to have:

- [x] automatic gradient
- [x] gradient descent
- [x] linear regression
- [x] logistic regression
- [x] useful math functions
- [x] multi-layer network
- [ ] utilities for SGD

Here is a simple code example for linear regression:

```python
import numpy as np
import dllib
from dllib.loss import mse_loss
from dllib.optimizer import GradientDescent

# define data
N = 1000
w0 = np.array([1.0, 2.0])
b0 = np.array([4.0])

X = np.random.randn(N, 2)
Y = np.dot(X, w0) + b0

# define placeholder
data = dllib.Placeholder((N, 2))
data.feed(X)

label = dllib.Placeholder((N,))
label.feed(Y)

# define variable
w = dllib.Variable(np.array([0.0, 0.0]), name="w")
b = dllib.Variable(np.array([0.0]), name="b")

# define loss
pred = b + data @ w
loss = dllib.reduce_mean((pred-label)*(pred-label))
loss.check_shape()

optimizer = GradientDescent(loss, 0.01)
optimizer.optimize(500, verbose=True)

print("w is: ", w.forward())
print("b is: ", b.forward())
```

Output is:

```
[  0/500] loss: 21.799619
[ 50/500] loss: 2.675228
[100/500] loss: 0.330923
[150/500] loss: 0.041273
[200/500] loss: 0.005191
[250/500] loss: 0.000658
[300/500] loss: 0.000084
[350/500] loss: 0.000011
[400/500] loss: 0.000001
[450/500] loss: 0.000000
w is:  [1.00002293 1.99996664]
b is:  [3.99984924]
```

Also a code example for logistic regression:

```python
import numpy as np
import dllib
from dllib.optimizer import GradientDescent
from dllib.loss import logistic_loss_with_logits
import matplotlib.pyplot as plt

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

w = dllib.Variable(np.array([0.0, 0.0]), 'w')
b = dllib.Variable(np.array([0.0]), 'b')

t = data @ w + b
loss = logistic_loss_with_logits(t, label)

optimizer = GradientDescent(loss, 0.01)
optimizer.optimize(500)

pred = np.sign(t.forward())
acc = pred[((pred - Y) == 0)].size/(2*N)
print("accuracy is: {0}".format(acc))
```

Output is:

```
[  0/500] loss: 0.693147
[ 50/500] loss: 0.491806
[100/500] loss: 0.394983
[150/500] loss: 0.340265
[200/500] loss: 0.305361
[250/500] loss: 0.281173
[300/500] loss: 0.263402
[350/500] loss: 0.249776
[400/500] loss: 0.238984
[450/500] loss: 0.230216
accuracy is: 0.915
```