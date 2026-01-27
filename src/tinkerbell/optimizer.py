from tinkerbell.tensor import Tensor
from typing import Iterator
import numpy as np

class Optimizer:
    def __init__(self, parameters:Iterator[Tensor], learning_rate=0.01):
        self.parameters = []
        for param in parameters:
            if param.requires_grad:
                self.parameters.append(param)
        self.learning_rate = learning_rate
        
    def step(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

class SGD(Optimizer):
    def __init__(self, parameters:Iterator[Tensor], learning_rate=0.01):
        super().__init__(parameters, learning_rate)

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                
                param.data -= self.learning_rate * param.grad
                param.grad =  np.zeros_like(param.data)