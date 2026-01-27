from tinkerbell.tensor import Tensor
import numpy as np
class Loss:
    def __init__(self):
        self.out: Tensor|None = None

    def assert_same_shape(self, a, b):
        assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        self.assert_same_shape(y_true, y_pred)
        self.y_true = y_true
        self.y_pred = y_pred
        self.out =  self.forward(y_true, y_pred)
        return self.out

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        raise NotImplementedError("Forward method not implemented.")
    
    def backward(self):
        if self.out is None:
            raise ValueError("No forward pass has been computed.")
        self.out.grad = np.ones_like(self.out.data)
        self.out.backward()



class MSELoss(Loss):
    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return Tensor.mean((y_true - y_pred) ** 2)
