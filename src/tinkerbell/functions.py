from tinkerbell.tensor import Tensor
import numpy as np

def sigmoid(tensor: Tensor) -> Tensor:
    data = 1 / (1 + np.exp(-tensor.data))
    out = Tensor(data, (tensor,))

    def _backward():
        sigmoid_derivative = out.data * (1 - out.data)
        tensor.grad += out.grad * sigmoid_derivative

    out._backward = _backward
    return out

def relu(tensor: Tensor) -> Tensor:
    data = np.maximum(0, tensor.data)
    out = Tensor(data, (tensor,))

    def _backward():
        relu_derivative = (out.data > 0).astype(out.data.dtype)
        tensor.grad += out.grad * relu_derivative

    out._backward = _backward
    return out
    