import unittest
from mini_pytorch.tensor import Tensor
from mini_pytorch.autograd import add, mul, matmul

class TestTensorOperations(unittest.TestCase):
    def test_add(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = add(a, b)
        c.backward(np.array([1, 1, 1]))
        self.assertTrue((a.grad == 1).all())
        self.assertTrue((b.grad == 1).all())

    def test_mul(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = mul(a, b)
        c.backward(np.array([1, 1, 1]))
        self.assertTrue((a.grad == b.data).all())
        self.assertTrue((b.grad == a.data).all())

    def test_matmul(self):
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)
        c = matmul(a, b)
        c.backward(np.array([[1, 1], [1, 1]]))
        expected_a_grad = b.data
        expected_b_grad = a.data
        self.assertTrue((a.grad == expected_a_grad).all())
        self.assertTrue((b.grad == expected_b_grad).all())

if __name__ == '__main__':
    unittest.main()
