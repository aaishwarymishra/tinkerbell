from tinkerbell.legacy.loss import Loss
from tinkerbell.legacy.optimizer import Optimizer
from tqdm.auto import tqdm
import numpy as np
from tinkerbell.legacy.layers import Layer

class Model:
    def __init__(self,layers: list[Layer] | None):
        self.layers: list[Layer] = layers if layers is not None else []
        self.loss: Loss | None = None
        self.optimizer: Optimizer | None = None
        self._input: np.ndarray | None = None
        self._output: np.ndarray | None = None
        self.grad_input: np.ndarray | None = None
        self.grad_output: np.ndarray | None = None
        self.first_forward = True


    def add(self, layer: Layer):
        self.layers.append(layer)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        self._output = self.forward(x)
        if self.first_forward:
            self.first_forward = False
        return self._output

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self,loss_grad) -> np.ndarray:
        self.grad_output = loss_grad
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        self.grad_input = grad
        return self.grad_input

    def get_params(self):
        params = []
        for layer in self.layers:
            for p in layer.get_params():
                yield p
    def get_grad_params(self):
        grad_params = []
        for layer in self.layers:
            for gp in layer.get_grad_params():
                yield gp

    def trainer(self, loss: Loss, optimizer:Optimizer):
        if not isinstance(loss, Loss):
            raise ValueError("Invalid loss function provided.")
        if not isinstance(optimizer, Optimizer):
            raise ValueError("Invalid optimizer provided.")
        self.loss = loss
        self.optimizer = optimizer

    def assert_same_shape(self, a, b):
        assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}" 
    
    def fit(self, x: np.ndarray, y: np.ndarray,
        validation_data: tuple[np.ndarray,np.ndarray]|None =None,
        epochs: int = 1,
        batch_size: int = 32):

        if self.loss is None or self.optimizer is None:
            raise ValueError("Model not compiled. Please set loss and optimizer before training.")

        assert x.shape[0] == y.shape[0], "Mismatched number of samples between x and y."
        if validation_data is not None:
            val_x, val_y = validation_data
            assert val_x.shape[0] == val_y.shape[0], "Mismatched number of samples between validation x and y."
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            num_samples = x.shape[0]
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start_idx in tqdm(range(0, num_samples, batch_size), desc="Training", leave=False):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                batch_x = x[batch_indices]
                batch_y = y[batch_indices]

                predictions = self(batch_x)
                loss_value = self.loss(batch_y, predictions)

                self.backward(self.loss.backward())


                self.optimizer.step(self.get_grad_params())

            if validation_data is not None:
                val_predictions = self(val_x)
                val_loss_value = self.loss(val_predictions, val_y)
                print(f"Validation Loss: {val_loss_value:.4f}")