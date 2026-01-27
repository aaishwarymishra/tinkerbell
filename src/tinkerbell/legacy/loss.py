import numpy as np

class Loss:
    def __init__(self):
        pass

    def assert_same_shape(self, a, b):
        assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        self.assert_same_shape(y_true, y_pred)
        self.y_true = y_true
        self.y_pred = y_pred
        self.out =  self.forward(y_true, y_pred)
        return self.out

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError("Forward method not implemented.")

    def backward(self) -> np.ndarray:
        self._grad_input = self.input_grad()
        self.assert_same_shape(self._grad_input, self.y_pred)
        return self._grad_input

    def input_grad(self) -> np.ndarray:
        raise NotImplementedError("Backward method not implemented.")
        

class MSELoss(Loss):
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    def input_grad(self) -> np.ndarray:
        n_samples = self.y_true.shape[0]
        return -2 * (self.y_true - self.y_pred) / n_samples