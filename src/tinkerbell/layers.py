from tinkerbell.tensor import Tensor, Parameter
from tinkerbell.functions import sigmoid, relu
from tinkerbell.optimizer import Optimizer
from tinkerbell.loss import Loss
import hashlib
import numpy as np
from tqdm.auto import tqdm

class Layer:
    def __init__(self):
        self._parameters = {}
        self._layers = {}
        self.input_shape = None
        self.output_shape = None
        self.first_forward = True
        self.name = self.__class__.__name__

    def __call__(self, input: Tensor) -> Tensor:
        self.input_shape = input.shape
        out = self.forward(input)
        self.output_shape = out.shape
        if (hasattr(self, 'first_forward') and self.first_forward) or (not hasattr(self, 'first_forward')):
            self.build()
            self.first_forward = False
        return out

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError("Forward method not implemented.")

    def build(self):
        for key, val in self.__dict__.items():
            if isinstance(val, Parameter):
                self.add_parameter(key, val)
            elif isinstance(val, Layer):
                self.add_layer(key, val)
            elif isinstance(val, list) and all(isinstance(item, Layer) for item in val):
                for idx, layer in enumerate(val):
                    self.add_layer(f"{key}_{idx}", layer)
            elif isinstance(val, list) and all(isinstance(item, Parameter) for item in val):
                for idx, param in enumerate(val):
                    self.add_parameter(f"{key}_{idx}", param)
        
        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__

    def parameters(self):
        for name, param in self._parameters.items():
            yield param
        for name, layer in self._layers.items():
            yield from layer.parameters()

    def add_parameter(self, name, param):
        if not hasattr(self, '_parameters'):
            self._parameters = {}
        param_name = f"{self.__class__.__name__}_{name}_{hashlib.md5(str(id(param)).encode()).hexdigest()[:6]}"
        self._parameters[param_name] = param

    def add_layer(self, name, layer):
        self._layers[name] = layer

    def __repr__(self):
        return f"Layer(name={self.name}, parameters={list(self._parameters.keys())}, layers={list(self._layers.keys())})"

class Dense(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = Parameter(Tensor.randn((input_dim, output_dim), requires_grad=True))
        self.bias = Parameter(Tensor.randn((output_dim,), requires_grad=True))

    def forward(self, input: Tensor) -> Tensor:
        return Tensor.matmul(input, self.weights) + self.bias

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return sigmoid(input)

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return relu(input)

class Sequential(Layer):

    def __init__(self,layers: list[Layer] | None):
        super().__init__()  
        self.layers: list[Layer] = layers if layers is not None else []

    def build(self):
        pass

    def forward(self, input:Tensor) -> Tensor:
        out = input
        for layer in self.layers:
            out = layer(out)
        return out


    def trainer(self, loss: Loss, optimizer:Optimizer):
        if not isinstance(loss, Loss):
            raise ValueError("Invalid loss function provided.")
        if not isinstance(optimizer, Optimizer):
            raise ValueError("Invalid optimizer provided.")
        self.loss = loss
        self.optimizer = optimizer

    def assert_same_shape(self, a, b):
        assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}" 
    
    def fit(self, x: Tensor|np.ndarray, y: Tensor|np.ndarray,
        validation_data: tuple[Tensor|np.ndarray,Tensor|np.ndarray]|None =None,
        epochs: int = 1,
        batch_size: int = 32):

        if self.loss is None or self.optimizer is None:
            raise ValueError("Model not compiled. Please set loss and optimizer before training.")

        assert x.shape[0] == y.shape[0], "Mismatched number of samples between x and y."
        if validation_data is not None:
            val_x, val_y = validation_data
            assert val_x.shape[0] == val_y.shape[0], "Mismatched number of samples between validation x and y."

        if isinstance(x, np.ndarray):
            x = Tensor(x)
        if isinstance(y, np.ndarray):
            y = Tensor(y)
        
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

                self.loss.backward()
                self.optimizer.step()

            if validation_data is not None:
                if isinstance(val_x, np.ndarray):
                    val_x = Tensor(val_x)
                if isinstance(val_y, np.ndarray):
                    val_y = Tensor(val_y)
                val_predictions = self(val_x)
                val_loss_value = self.loss(val_predictions, val_y)
                print(f"Validation Loss: {val_loss_value:.4f}")