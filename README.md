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
- [x] useful math functions

Here is a simple code for linear regression:

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
optimizer.train(500, verbose=True)

print("w is: ", w.forward())
print("b is: ", b.forward())
```

Output is:

```
[    0/500] loss: 21.017926
[   50/500] loss: 2.782524
[  100/500] loss: 0.371020
[  150/500] loss: 0.049847
[  200/500] loss: 0.006750
[  250/500] loss: 0.000921
[  300/500] loss: 0.000127
[  350/500] loss: 0.000018
[  400/500] loss: 0.000002
[  450/500] loss: 0.000000
w is:  [1.00003636 1.99986063]
b is:  [3.99982599]
```