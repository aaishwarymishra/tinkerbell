import numpy as np
from typing import Iterator

class Optimizer:
    def __init__(self,params:Iterator[np.ndarray]|list[np.ndarray], learning_rate: float = 0.01):
        self.params = list(params)
        self.learning_rate = learning_rate

    def step(self, grad_params: Iterator[np.ndarray]):
        pass

class SGD(Optimizer):
    def __init__(self, params:Iterator[np.ndarray], learning_rate: float = 0.01):
        super().__init__(params,learning_rate)

    def step(self,grad_params: Iterator[np.ndarray]):
        for param, grad_param in zip(self.params, grad_params):
            param -= self.learning_rate * grad_param