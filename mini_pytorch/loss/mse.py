from ..tensor import Tensor
from ..autograd import Function
import numpy as np

class MSELoss(Function):
    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target
        loss = np.mean((prediction.data - target.data) ** 2)
        self.output = Tensor(loss, requires_grad=True)
        return self.output

    def backward(self, grad_output):
        grad = (2 * (self.prediction.data - self.target.data)) / self.target.data.size
        return grad_output * grad, None # 返回 prediction 的梯度和 target 的梯度 (None)

def mse_loss(prediction, target):
    return MSELoss()(prediction, target)
