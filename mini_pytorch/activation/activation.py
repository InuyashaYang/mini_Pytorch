from ..autograd import Tensor
import numpy as np

class ReLU:
    def __init__(self):
        self.output = None

    def forward(self, x):
        print(x,type(x))
        self.input = x
        self.output = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad, _children=(x,), _op=None)
        return self.output

    def backward(self, grad_output):
        grad_input = grad_output * (self.input.data > 0)
        self.input.grad = grad_input if self.input.grad is None else self.input.grad + grad_input
