from ..tensor import Tensor
from ..autograd import Function
import numpy as np

class ReLU(Function):
    def forward(self, x):
        self.x = x
        self.output = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
        return self.output

    def backward(self, grad_output):
        grad = grad_output * (self.x.data > 0).astype(float)
        self.x.backward(grad)

def relu(x):
    return ReLU()(x)
