"""Optimization module"""
import needle as ndl
import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def step(self):
        """Update parameters using their accumulated gradients."""

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad.data

            if self.weight_decay != 0:
                grad += self.weight_decay * param.data

            if param not in self.u:
                self.u[param] = grad * 0

            self.u[param] = (self.momentum * self.u[param] + (1 - self.momentum) * grad).data
            param.data = (param.data - self.lr * self.u[param]).data

    def clip_grad_norm(self, max_norm=0.25):
        """Scale gradients in place when their global L2 norm exceeds max_norm."""
        if max_norm < 0:
            raise ValueError("max_norm must be nonnegative")
        grads = [param.grad for param in self.params if param.grad is not None]
        total_norm = np.sqrt(sum(float((grad.numpy() ** 2).sum()) for grad in grads))
        if total_norm > max_norm:
            scale = np.float32(max_norm / (total_norm + 1e-6))
            for grad in grads:
                grad.data = (grad.data * scale).data
        return total_norm


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1

        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad.data

            grad = grad + self.weight_decay * param.data

            if param not in self.m:
                self.m[param] = grad * 0
                self.v[param] = grad * 0

            self.m[param] = (self.beta1 * self.m[param] + (1 - self.beta1) * grad).data
            self.v[param] = (self.beta2 * self.v[param] + (1 - self.beta2) * grad ** 2).data
            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)

            param.data = (param.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)).data
