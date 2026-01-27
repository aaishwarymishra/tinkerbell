from tinkerbell.tensor import Tensor
import numpy as np

def Sigmoid(tensor: Tensor) -> Tensor:
    data = 1 / (1 + np.exp(-tensor.data))
    out = Tensor(data, (tensor,))

    def _backward():
        sigmoid_derivative = out.data * (1 - out.data)
        tensor.grad += out.grad * sigmoid_derivative

    out._backward = _backward
    return out