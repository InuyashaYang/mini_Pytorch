from ..autograd import Tensor
import numpy as np
from mini_pytorch.activation.activation import ReLU
from mini_pytorch.loss.loss import CrossEntropyLoss

class Linear:
    def __init__(self, input_size, output_size):
        self.weight = Tensor(np.random.randn(input_size, output_size) * 0.01, requires_grad=True)
        self.bias = Tensor(np.zeros((1, output_size)), requires_grad=True)

    def forward(self, x):
        return x @ self.weight + self.bias
    

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        input_size: 输入特征数
        hidden_sizes: 隐藏层大小的列表
        output_size: 输出类别数
        """
        self.layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes)-1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                self.layers.append(ReLU())
        # 损失函数
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        # 反向传播
        for layer in reversed(self.layers):
            if isinstance(layer, ReLU):
                layer.backward(layer.output.grad)
            elif isinstance(layer, Linear):
                layer.backward(layer.output.grad)

    def parameters(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                params.extend([layer.weight, layer.bias])
        return params