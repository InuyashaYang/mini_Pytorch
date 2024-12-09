import numpy as np

class Tensor:
    def __init__(self, data, dtype=float, requires_grad=False, _children=(), _op=None):
        self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def T(self):
        return Tensor(self.data.T, requires_grad=self.requires_grad, _children=(self,), _op=None)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Add().forward(self, other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Mul().forward(self, other)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return MatMul().forward(self, other)

    def __pow__(self, other):
        return Pow().forward(self, other)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Sub().forward(self, other)

    def sum(self):
        return Sum().forward(self)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Div().forward(self, other)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rtruediv__ = __truediv__

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node._op:
                node._op.backward(node.grad)


class Operation:
    def forward(self, *tensors):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Add(Operation):
    def forward(self, x, y):
        self.x = x
        self.y = y
        return Tensor(x.data + y.data, requires_grad=x.requires_grad or y.requires_grad, _children=(x, y), _op=self)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = grad
            if self.x.data.shape != grad.shape:
                grad_x = grad.sum(axis=0, keepdims=True)
            self.x.grad = grad_x if self.x.grad is None else self.x.grad + grad_x
        if self.y.requires_grad:
            grad_y = grad
            if self.y.data.shape != grad.shape:
                grad_y = grad.sum(axis=0, keepdims=True)
            self.y.grad = grad_y if self.y.grad is None else self.y.grad + grad_y


class Mul(Operation):
    def forward(self, x, y):
        self.x = x
        self.y = y
        return Tensor(x.data * y.data, requires_grad=x.requires_grad or y.requires_grad, _children=(x, y), _op=self)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = self.y.data * grad
            if self.x.data.shape != grad.shape:
                grad_x = grad_x.sum(axis=0, keepdims=True)
            self.x.grad = grad_x if self.x.grad is None else self.x.grad + grad_x
        if self.y.requires_grad:
            grad_y = self.x.data * grad
            if self.y.data.shape != grad.shape:
                grad_y = grad_y.sum(axis=0, keepdims=True)
            self.y.grad = grad_y if self.y.grad is None else self.y.grad + grad_y


class MatMul(Operation):
    def forward(self, x, y):
        self.x = x
        self.y = y
        return Tensor(x.data @ y.data, requires_grad=x.requires_grad or y.requires_grad, _children=(x, y), _op=self)

    def backward(self, grad):
        if self.x.requires_grad:
            self.x.grad = grad @ self.y.data.T if self.x.grad is None else self.x.grad + grad @ self.y.data.T
        if self.y.requires_grad:
            self.y.grad = self.x.data.T @ grad if self.y.grad is None else self.y.grad + self.x.data.T @ grad


class Pow(Operation):
    def forward(self, x, other):
        assert isinstance(other, (int, float))
        self.x = x
        self.other = other
        return Tensor(x.data**other, requires_grad=x.requires_grad, _children=(x,), _op=self)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = (self.other * self.x.data**(self.other - 1)) * grad
            self.x.grad = grad_x if self.x.grad is None else self.x.grad + grad_x


class Sub(Operation):
    def forward(self, x, y):
        self.x = x
        self.y = y
        return Tensor(x.data - y.data, requires_grad=x.requires_grad or y.requires_grad, _children=(x, y), _op=self)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = grad
            if self.x.data.shape != grad.shape:
                grad_x = grad.sum(axis=0, keepdims=True)
            self.x.grad = grad_x if self.x.grad is None else self.x.grad + grad_x

        if self.y.requires_grad:
            grad_y = -grad
            if self.y.data.shape != grad.shape:
                grad_y = grad_y.sum(axis=0, keepdims=True)
            self.y.grad = grad_y if self.y.grad is None else self.y.grad + grad_y


class Sum(Operation):
    def forward(self, x):
        self.x = x
        return Tensor(np.sum(x.data), requires_grad=x.requires_grad, _children=(x,), _op=self)

    def backward(self, grad):
        if self.x.requires_grad:
            self.x.grad = np.ones_like(self.x.data) * grad if self.x.grad is None else self.x.grad + np.ones_like(self.x.data) * grad


class Div(Operation):
    def forward(self, x, y):
        self.x = x
        self.y = y
        return Tensor(x.data / y.data, requires_grad=x.requires_grad or y.requires_grad, _children=(x, y), _op=self)

    def backward(self, grad):
        if self.x.requires_grad:
            grad_x = (1.0 / self.y.data) * grad
            if self.x.data.shape != grad.shape:
                grad_x = grad_x.sum(axis=0, keepdims=True)
            self.x.grad = grad_x if self.x.grad is None else self.x.grad + grad_x
        if self.y.requires_grad:
            grad_y = (-self.x.data / (self.y.data**2)) * grad
            if self.y.data.shape != grad.shape:
                grad_y = grad_y.sum(axis=0, keepdims=True)
            self.y.grad = grad_y if self.y.grad is None else self.y.grad + grad_y

