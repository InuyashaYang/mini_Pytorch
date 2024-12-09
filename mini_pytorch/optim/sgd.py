class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters  # 参数列表
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None