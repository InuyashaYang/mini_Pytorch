import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self.depends_on = []  # 存储依赖项，用于反向传播

    def set_creator(self, func):
        self.depends_on.append(func) # 将依赖函数添加到列表

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            print('no grad now\n')
            if self.data.size == 1:
                grad = np.array(1.0)
            else:
                grad = np.ones_like(self.data)
                print('grad filled')

        if self.grad is None:  # 避免重复累加梯度
            self.grad = grad
        else:
            self.grad += grad  # 累加梯度


        for func in self.depends_on: # 遍历依赖项并反向传播
            if func: #检查func是否为None
                backward_grad = func.backward(self, grad)
                if backward_grad is not None: # 处理backward函数返回None的情况
                    # 这里可能需要根据你的autograd实现进行调整，例如累加梯度等
                    pass

    def size(self):
        return self.data.shape

    def __add__(self, other):
        from .autograd import add  # 局部导入
        return add(self, other)

    def __radd__(self, other): # 支持反向加法，例如 3 + Tensor
        return self + other

    def __mul__(self, other):
        from .autograd import mul  # 局部导入
        return mul(self, other)

    def __rmul__(self, other): # 支持反向乘法
        return self * other

    def __matmul__(self, other):
        from .autograd import matmul  # 局部导入
        return matmul(self, other)


    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, grad={self.grad})" # 显示梯度信息
