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
- [ ] softmax regression
- [ ] utilities for SGD
- [ ] different activation functions
- [ ] useful math functions

Here is a simple code for linear regression:

```python
import numpy as np
from ops import *
from optimizer import GradientDescent

N = 1000
# target value
w0 = np.array([1.0, 2.0])
b0 = np.array([4.0])

# random data
X = np.random.randn(N, 2)
Y = np.dot(X, w0) + b0

# define placeholder to build graph
data = Placeholder((N, 2))
data.feed(X)
label = Placeholder((N,))
label.feed(Y)

# trainable parameters
w = Variable(np.array([0.0, 0.0]), name="w")
b = Variable(np.array([0.8]), name="b")

# define computation graph
pred = VMulOp(data, w)
pred = pred + b
m = pred - label
loss = VMulOp(m, m)
loss = loss*(1/N)

optimizer = GradientDescent(loss, 0.01)
optimizer.train(500)

print(w.forward())
print(b.forward())
```

Output is:

```
round:    0/500, loss: 16.078551
round:   50/500, loss: 1.906219
round:  100/500, loss: 0.228830
round:  150/500, loss: 0.027848
round:  200/500, loss: 0.003439
round:  250/500, loss: 0.000431
round:  300/500, loss: 0.000055
round:  350/500, loss: 0.000007
round:  400/500, loss: 0.000001
round:  450/500, loss: 0.000000
w is:  [1.00003137 1.99996939]
b is:  [3.99987718]
```