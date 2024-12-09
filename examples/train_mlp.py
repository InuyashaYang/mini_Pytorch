from mini_pytorch.autograd import Tensor
from mini_pytorch.nn.modules import MLP,Linear
from mini_pytorch.optim.sgd import SGD
import numpy as np
from matplotlib import pyplot as plt


# 训练数据
X = np.array([[1, 2], [3, 4]])
y = np.array([[5], [11]])

# 转换为 Tensor
X_tensor = Tensor(X)
y_tensor = Tensor(y)

# 定义模型：输入层 2 个神经元，隐藏层 3 个神经元，输出层 1 个神经元
input_size = 2
hidden_size = 3
output_size = 1

model1 = Linear(input_size, hidden_size)
model2 = Linear(hidden_size, output_size)

learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    # 前向传播
    hidden_output = model1.forward(X_tensor)
    output = model2.forward(hidden_output)

    # 计算损失
    loss = ((output - y_tensor)**2).sum() / y_tensor.data.size

    # 执行反向传播
    loss.backward()

    # 更新参数
    model1.weight.data -= learning_rate * model1.weight.grad
    model1.bias.data -= learning_rate * model1.bias.grad
    model2.weight.data -= learning_rate * model2.weight.grad
    model2.bias.data -= learning_rate * model2.bias.grad

    # 清零梯度，防止梯度累加
    model1.weight.grad = None
    model1.bias.grad = None
    model2.weight.grad = None
    model2.bias.grad = None

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")  # 访问 loss.data 进行打印

print("Final Weight1:", model1.weight.data)
print("Final Bias1:", model1.bias.data)
print("Final Weight2:", model2.weight.data)
print("Final Bias2:", model2.bias.data)