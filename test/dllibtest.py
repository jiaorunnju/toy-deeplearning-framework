import unittest
import dllib as d
import numpy as np
from numpy import array, ndarray


def arr_equal(x: ndarray, y: ndarray):
    if not isinstance(x, ndarray) or not isinstance(y, ndarray):
        raise RuntimeError
    return (x == y).all()


class TestAdd(unittest.TestCase):

    def test_add(self):
        x = d.Variable(array([2.0, 3.0]), 'x')
        y = d.Variable(array([6.0, 7.0]), 'y')
        z = x + y
        t = z.forward()
        self.assertTrue(arr_equal(t, array([8, 10])))
        z.backward(array([2.0, 2.0]))
        self.assertTrue(arr_equal(x.grad, array([2, 2])))
        self.assertTrue(arr_equal(y.grad, array([2, 2])))

    def test_add_broadcast1(self):
        x = d.Variable(array([2.0, 3.0]), 'x')
        y = d.Variable(array([6.0]), 'y')
        z = x + y
        t = z.forward()
        self.assertTrue(arr_equal(t, array([8, 9])))
        z.backward(array([2.0, 1.0]))
        self.assertTrue(arr_equal(x.grad, array([2, 1])))
        self.assertTrue(arr_equal(y.grad, array([3])))

    def test_add_broadcast2(self):
        x = d.Variable(array([[2.0, 3.0], [5.0, 6.0]]), 'x')
        y = d.Variable(array([6.0, 6.0]), 'y')
        z = x + y
        t = z.forward()
        self.assertTrue(arr_equal(t, array([[8, 9], [11, 12]])))
        z.backward(array([[1.0, 1.0], [2, 1]]))
        self.assertTrue(arr_equal(x.grad, array([[1.0, 1.0], [2, 1]])))
        self.assertTrue(arr_equal(y.grad, array([3, 2])))

    def test_add_broadcast3(self):
        x = d.Variable(array([2.0, 3.0]), 'x')
        y = d.Variable(array([6.0]), 'y')
        z = y + x
        t = z.forward()
        self.assertTrue(arr_equal(t, array([8, 9])))
        z.backward(array([1.0, 1.0]))
        self.assertTrue(arr_equal(x.grad, array([1, 1])))
        self.assertTrue(arr_equal(y.grad, array([2])))

    def test_add_broadcast4(self):
        x = d.Variable(array([[2.0, 3.0], [5.0, 6.0]]), 'x')
        y = d.Variable(array([6.0, 6.0]), 'y')
        z = y + x
        t = z.forward()
        self.assertTrue(arr_equal(t, array([[8, 9], [11, 12]])))
        z.backward(array([[1.0, 1.0], [1, 1]]))
        self.assertTrue(arr_equal(x.grad, array([[1.0, 1.0], [1, 1]])))
        self.assertTrue(arr_equal(y.grad, array([2, 2])))

    def test_add_broadcast5(self):
        x = d.Variable(array([[2.0, 3.0], [5.0, 6.0]]), 'x')
        y = d.Variable(array([6.0]), 'y')
        z = x + y
        t = z.forward()
        self.assertTrue(arr_equal(t, array([[8, 9], [11, 12]])))
        z.backward(array([[1.0, 1.0], [2, 1]]))
        self.assertTrue(arr_equal(x.grad, array([[1.0, 1.0], [2, 1]])))
        self.assertTrue(arr_equal(y.grad, array([5])))

    def test_add_broadcast6(self):
        x = d.Variable(array([[2.0, 3.0], [5.0, 6.0]]), 'x')
        y = d.Variable(array([6.0]), 'y')
        z = y + x
        t = z.forward()
        self.assertTrue(arr_equal(t, array([[8, 9], [11, 12]])))
        z.backward(array([[1.0, 1.0], [2, 1]]))
        self.assertTrue(arr_equal(x.grad, array([[1.0, 1.0], [2, 1]])))
        self.assertTrue(arr_equal(y.grad, array([5])))


class TestSub(unittest.TestCase):

    def test_sub(self):
        x = d.Variable(array([2.0, 3.0]), 'x')
        y = d.Variable(array([6.0, 7.0]), 'y')
        z = x - y
        t = z.forward()
        self.assertTrue(arr_equal(t, array([-4, -4])))
        z.backward(array([2.0, 1.0]))
        self.assertTrue(arr_equal(x.grad, array([2, 1])))
        self.assertTrue(arr_equal(y.grad, -array([2, 1])))

    def test_sub_broadcast1(self):
        x = d.Variable(array([2.0, 3.0]), 'x')
        y = d.Variable(array([6.0]), 'y')
        z = x - y
        t = z.forward()
        self.assertTrue(arr_equal(t, array([-4, -3])))
        z.backward(array([1.0, 2.0]))
        self.assertTrue(arr_equal(x.grad, array([1, 2])))
        self.assertTrue(arr_equal(y.grad, array([-3])))

    def test_sub_broadcast2(self):
        x = d.Variable(array([[2.0, 3.0], [5.0, 6.0]]), 'x')
        y = d.Variable(array([6.0, 6.0]), 'y')
        z = x - y
        t = z.forward()
        self.assertTrue(arr_equal(t, array([[-4, -3], [-1, 0]])))
        z.backward(array([[1.0, 1.0], [1, 1]]))
        self.assertTrue(arr_equal(x.grad, array([[1.0, 1.0], [1, 1]])))
        self.assertTrue(arr_equal(y.grad, -array([2, 2])))

    def test_sub_broadcast3(self):
        x = d.Variable(array([2.0, 3.0]), 'x')
        y = d.Variable(array([6.0]), 'y')
        z = y - x
        t = z.forward()
        self.assertTrue(arr_equal(t, array([4, 3])))
        z.backward(array([2.0, 1.0]))
        self.assertTrue(arr_equal(x.grad, -array([2, 1])))
        self.assertTrue(arr_equal(y.grad, array([3])))

    def test_sub_broadcast4(self):
        x = d.Variable(array([[2.0, 3.0], [5.0, 6.0]]), 'x')
        y = d.Variable(array([6.0, 6.0]), 'y')
        z = y - x
        t = z.forward()
        self.assertTrue(arr_equal(t, array([[4, 3], [1, 0]])))
        z.backward(array([[1.0, 1.0], [1, 1]]))
        self.assertTrue(arr_equal(x.grad, -array([[1.0, 1.0], [1, 1]])))
        self.assertTrue(arr_equal(y.grad, array([2, 2])))

    def test_sub_broadcast5(self):
        x = d.Variable(array([[2.0, 3.0], [5.0, 6.0]]), 'x')
        y = d.Variable(array([6.0]), 'y')
        z = y - x
        t = z.forward()
        self.assertTrue(arr_equal(t, array([[4, 3], [1, 0]])))
        z.backward(array([[1.0, 1.0], [2, 1]]))
        self.assertTrue(arr_equal(x.grad, -array([[1.0, 1.0], [2, 1]])))
        self.assertTrue(arr_equal(y.grad, array([5])))

    def test_sub_broadcast6(self):
        x = d.Variable(array([[2.0, 3.0], [5.0, 6.0]]), 'x')
        y = d.Variable(array([6.0]), 'y')
        z = x - y
        t = z.forward()
        self.assertTrue(arr_equal(t, array([[-4, -3], [-1, 0]])))
        z.backward(array([[1.0, 1.0], [1, 1]]))
        self.assertTrue(arr_equal(x.grad, array([[1.0, 1.0], [1, 1]])))
        self.assertTrue(arr_equal(y.grad, -array([4])))


class TestMul(unittest.TestCase):

    def test_mul1(self):
        x = d.Variable(array([3, 1]), 'x')
        y = d.Variable(array([4, 6]), 'y')
        z = x * y
        t = z.forward()
        z.backward(array([1, 1]))
        self.assertTrue(arr_equal(t, array([12, 6])))
        self.assertTrue(arr_equal(x.grad, array([4, 6])))
        self.assertTrue(arr_equal(y.grad, array([3, 1])))

    def test_mul2(self):
        x = d.Variable(array([3, 1]), 'x')
        y = d.Variable(array([4, 6]), 'y')
        z = y * x
        t = z.forward()
        z.backward(array([1, 1]))
        self.assertTrue(arr_equal(t, array([12, 6])))
        self.assertTrue(arr_equal(x.grad, array([4, 6])))
        self.assertTrue(arr_equal(y.grad, array([3, 1])))

    def test_mul3(self):
        x = d.Variable(array([3]), 'x')
        y = d.Variable(array([4]), 'y')
        z = y * x
        t = z.forward()
        z.backward(array([1]))
        self.assertTrue(arr_equal(t, array([12])))
        self.assertTrue(arr_equal(x.grad, array([4])))
        self.assertTrue(arr_equal(y.grad, array([3])))

    def test_mul4(self):
        x = d.Variable(array([3, 1]), 'x')
        y = d.Variable(array([4]), 'y')
        z = y * x
        t = z.forward()
        z.backward(array([1, 2]))
        self.assertTrue(arr_equal(t, array([12, 4])))
        self.assertTrue(arr_equal(x.grad, array([4, 8])))
        self.assertTrue(arr_equal(y.grad, array([5])))

    def test_mul5(self):
        x = d.Variable(array([[3, 1], [2, 1]]), 'x')
        y = d.Variable(array([4]), 'y')
        z = y * x
        t = z.forward()
        z.backward(array([[1, 2], [1, 2]]))
        self.assertTrue(arr_equal(t, array([[12, 4], [8, 4]])))
        self.assertTrue(arr_equal(x.grad, array([[4, 8], [4, 8]])))
        self.assertTrue(arr_equal(y.grad, array([9])))

    def test_mul6(self):
        x = d.Variable(array([[3, 1], [2, 1]]), 'x')
        y = d.Variable(array([4]), 'y')
        z = x * y
        t = z.forward()
        z.backward(array([[1, 2], [1, 2]]))
        self.assertTrue(arr_equal(t, array([[12, 4], [8, 4]])))
        self.assertTrue(arr_equal(x.grad, array([[4, 8], [4, 8]])))
        self.assertTrue(arr_equal(y.grad, array([9])))

    def test_mul7(self):
        x = d.Variable(array([[3, 1], [2, 1]]), 'x')
        y = d.Variable(array([4, 5]), 'y')
        z = x * y
        t = z.forward()
        z.backward(array([[1, 2], [1, 2]]))
        self.assertTrue(arr_equal(t, array([[12, 5], [8, 5]])))
        self.assertTrue(arr_equal(x.grad, array([[4, 10], [4, 10]])))
        self.assertTrue(arr_equal(y.grad, array([5, 4])))

    def test_mul8(self):
        x = d.Variable(array([[3, 1], [2, 1]]), 'x')
        y = d.Variable(array([4, 5]), 'y')
        z = y * x
        t = z.forward()
        z.backward(array([[1, 2], [1, 2]]))
        self.assertTrue(arr_equal(t, array([[12, 5], [8, 5]])))
        self.assertTrue(arr_equal(x.grad, array([[4, 10], [4, 10]])))
        self.assertTrue(arr_equal(y.grad, array([5, 4])))


class TestMMul(unittest.TestCase):

    def test_mmul1(self):
        w_ = np.array([[2, 3, 4], [1, 2, 3]])
        b_ = np.array([3, 4, 5])
        w = d.Variable(w_, 'w')
        b = d.Variable(b_, 'b')
        z = w @ b
        t = z.forward()
        self.assertTrue(arr_equal(t, array([38, 26])))
        z.backward(array([2, 1]))
        self.assertTrue(arr_equal(w.grad, array([[6, 8, 10], [3, 4, 5]])))
        self.assertTrue(arr_equal(b.grad, array([5, 8, 11])))

    def test_mmul2(self):
        w_ = np.array([2, 3, 4])
        b_ = np.array([3, 4, 5])
        w = d.Variable(w_, 'w')
        b = d.Variable(b_, 'b')
        z = w @ b
        t = z.forward()
        self.assertTrue(arr_equal(t, array(38)))
        z.backward(array([2]))
        self.assertTrue(arr_equal(w.grad, array([6, 8, 10])))
        self.assertTrue(arr_equal(b.grad, array([4, 6, 8])))


class TestDiv(unittest.TestCase):

    def test_div1(self):
        x = d.Variable(np.array([4]), 'x')
        y = d.Variable(np.array([2]), 'y')
        z = x / y
        t = z.forward()
        self.assertTrue(arr_equal(t, array([2])))
        z.backward(np.array([2]))
        self.assertTrue(arr_equal(x.grad, array([1])))
        self.assertTrue(arr_equal(y.grad, array([-2])))

    def test_div2(self):
        x = d.Variable(np.array([4, 8]), 'x')
        y = d.Variable(np.array([2, 4]), 'y')
        z = y / x
        t = z.forward()
        self.assertTrue(arr_equal(t, array([0.5, 0.5])))
        z.backward(np.array([2, 1]))
        self.assertTrue(arr_equal(x.grad, array([-0.25, -1 / 16])))
        self.assertTrue(arr_equal(y.grad, array([0.5, 0.125])))

    def test_div3(self):
        x = d.Variable(np.array([4, 2]), 'x')
        y = d.Variable(np.array([2]), 'y')
        z = x / y
        t = z.forward()
        self.assertTrue(arr_equal(t, array([2, 1])))
        z.backward(np.array([2, 1]))
        self.assertTrue(arr_equal(x.grad, array([1, 0.5])))
        self.assertTrue(arr_equal(y.grad, array([-2.5])))

    def test_div4(self):
        y = d.Variable(np.array([4, 2]), 'x')
        x = d.Variable(np.array([2]), 'y')
        z = x / y
        t = z.forward()
        self.assertTrue(arr_equal(t, array([0.5, 1])))
        z.backward(np.array([2, 1]))
        self.assertTrue(arr_equal(x.grad, array([1])))
        self.assertTrue(arr_equal(y.grad, array([-0.25, -0.5])))

    def test_div5(self):
        x = d.Variable(np.array([[4, 2], [4, 2]]), 'x')
        y = d.Variable(np.array([2]), 'y')
        z = x / y
        t = z.forward()
        self.assertTrue(arr_equal(t, array([[2, 1], [2, 1]])))
        z.backward(np.array([[2, 1], [2, 1]]))
        self.assertTrue(arr_equal(x.grad, array([[1, 0.5], [1, 0.5]])))
        self.assertTrue(arr_equal(y.grad, array([-5])))

    def test_div6(self):
        x = d.Variable(np.array([[4, 2], [4, 2]]), 'x')
        y = d.Variable(np.array([2]), 'y')
        z = y / x
        t = z.forward()
        self.assertTrue(arr_equal(t, array([[0.5, 1], [0.5, 1]])))
        z.backward(np.array([[2, 1], [2, 1]]))
        self.assertTrue(arr_equal(x.grad, array([[-0.25, -0.5], [-0.25, -0.5]])))
        self.assertTrue(arr_equal(y.grad, array([2])))

    def test_div7(self):
        x = d.Variable(np.array([[4, 2], [4, 2]]), 'x')
        y = d.Variable(np.array([2, 2]), 'y')
        z = x / y
        t = z.forward()
        self.assertTrue(arr_equal(t, array([[2, 1], [2, 1]])))
        z.backward(np.array([[2, 1], [2, 1]]))
        self.assertTrue(arr_equal(x.grad, array([[1, 0.5], [1, 0.5]])))
        self.assertTrue(arr_equal(y.grad, array([-4, -1])))

    def test_div8(self):
        x = d.Variable(np.array([[4, 2], [4, 2]]), 'x')
        y = d.Variable(np.array([2, 2]), 'y')
        z = y / x
        t = z.forward()
        self.assertTrue(arr_equal(t, array([[0.5, 1], [0.5, 1]])))
        z.backward(np.array([[2, 1], [2, 1]]))
        self.assertTrue(arr_equal(x.grad, array([[-0.25, -0.5], [-0.25, -0.5]])))
        self.assertTrue(arr_equal(y.grad, array([1, 1])))

    def test_div9(self):
        x = d.Variable(np.array([1, 2]), 'x')
        z = 2 / x
        t = z.forward()
        self.assertTrue(arr_equal(t, array([2, 1])))
        z.backward(np.array([1, 2]))
        self.assertTrue(arr_equal(x.grad, array([-2, -1])))


class TestMaxMin(unittest.TestCase):

    def test_max1(self):
        x = d.Variable(array([[1, 2], [-1, -2]]), 'x')
        y = d.Variable(array([[2, 1], [0, -3]]), 'y')
        z = d.max(x, y)
        t = z.forward()
        self.assertTrue(arr_equal(t, array([[2, 2], [0, -2]])))
        z.backward(array([[2, 2], [2, 2]]))
        self.assertTrue(arr_equal(x.grad, array([[0, 2], [0, 2]])))
        self.assertTrue(arr_equal(y.grad, array([[2, 0], [2, 0]])))

    def test_max2(self):
        x = d.Variable(array([[1, 2], [-1, -2]]), 'x')
        z = d.max(x, 0)
        t = z.forward()
        self.assertTrue(arr_equal(t, array([[1, 2], [0, 0]])))
        z.backward(array([[2, 2], [2, 2]]))
        self.assertTrue(arr_equal(x.grad, array([[2, 2], [0, 0]])))

    def test_min1(self):
        x = d.Variable(array([[1, 2], [-1, -2]]), 'x')
        y = d.Variable(array([[2, 1], [0, -3]]), 'y')
        z = d.min(x, y)
        t = z.forward()
        self.assertTrue(arr_equal(t, array([[1, 1], [-1, -3]])))
        z.backward(array([[2, 2], [2, 2]]))
        self.assertTrue(arr_equal(y.grad, array([[0, 2], [0, 2]])))
        self.assertTrue(arr_equal(x.grad, array([[2, 0], [2, 0]])))

    def test_min2(self):
        x = d.Variable(array([[1, 2], [-1, -2]]), 'x')
        z = d.min(0, x)
        t = z.forward()
        self.assertTrue(arr_equal(t, array([[0, 0], [-1, -2]])))
        z.backward(array([[2, 2], [2, 2]]))
        self.assertTrue(arr_equal(x.grad, array([[0, 0], [2, 2]])))


if __name__ == '__main__':
    unittest.main()
