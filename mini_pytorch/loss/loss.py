from ..autograd import Tensor
import numpy as np

class CrossEntropyLoss:
    def forward(self, logits, labels):
        """
        logits: 模型输出，未经过softmax，形状为 (batch_size, num_classes)
        labels: 真实标签，形状为 (batch_size,)
        """
        self.logits = logits
        self.labels = labels
        # 稳定的softmax计算
        exps = np.exp(logits.data - np.max(logits.data, axis=1, keepdims=True))
        self.softmax = exps / np.sum(exps, axis=1, keepdims=True)
        m = labels.data.shape[0]
        # 交叉熵损失
        loss = -np.log(self.softmax[np.arange(m), labels.data])  # shape: (batch_size,)
        loss = np.mean(loss)
        return Tensor(loss, requires_grad=True)

    def backward(self):
        """
        计算梯度
        """
        m = self.labels.data.shape[0]
        grad_logits = self.softmax.copy()
        grad_logits[np.arange(m), self.labels.data] -= 1
        grad_logits /= m
        self.logits.grad = grad_logits if self.logits.grad is None else self.logits.grad + grad_logits
