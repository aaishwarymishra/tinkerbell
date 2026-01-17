import numpy as np
from blocks.loss import Loss

class FunctionBlock:
    def __init__(self):
        pass

    def __call__(self, input):
        self._input = input
        self._output = self.forward(input)
        return self._output

    def forward(self, input):
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, grad_output):
        self.assert_same_shape(grad_output, self._output)
        self._grad_input = self.input_grad(grad_output)
        self.assert_same_shape(self._grad_input, self._input)
        return self._grad_input
        

    def input_grad(self, grad_output):
        raise NotImplementedError("Backward method not implemented.")

    def assert_same_shape(self, a, b):
        assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}" 


class LinearBlock(FunctionBlock):
    def __init__(self, param):
        super().__init__()
        self.param = param

    def backward(self, grad_output):
        self.assert_same_shape(grad_output, self._output)
        self._grad_input = self.input_grad(grad_output)
        self.assert_same_shape(self._grad_input, self._input)
        self._grad_param = self.param_grad(grad_output)
        self.assert_same_shape(self._grad_param, self.param)
        return self._grad_input

    def param_grad(self, grad_output):
        raise NotImplementedError("Parameter gradient method not implemented.")


class WeightMatrixBlock(LinearBlock):
    def __init__(self, weight: np.ndarray):
        super().__init__(weight)
        
    def forward(self, input: np.ndarray) -> np.ndarray:
        return np.matmul(input, self.param)

    def input_grad(self, grad_output: np.ndarray) -> np.ndarray:
        if self.param.ndim > 2:
            axis = list(range(grad_output.ndim - 2))
            axis.extend([grad_output.ndim - 1, grad_output.ndim - 2])
        else:
            axis = [1, 0]
        return np.matmul(grad_output, np.transpose(self.param,tuple(axis)))

    def param_grad(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input.ndim > 2:
            _in = np.reshape(self._input, (-1, self._input.shape[-1]))
            grad_out = np.reshape(grad_output, (-1, grad_output.shape[-1]))
            return np.matmul(np.transpose(_in), grad_out)    

        return np.matmul(np.transpose(self._input), grad_output)

class BiasBlock(LinearBlock):
    def __init__(self, bias: np.ndarray):
        super().__init__(bias)

    def forward(self, input: np.ndarray) -> np.ndarray:
        return input + self.param

    def input_grad(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output

    def param_grad(self, grad_output: np.ndarray) -> np.ndarray:
        return np.sum(grad_output, axis=0, keepdims=True)

class Layer:
    def __init__(self,output_dim:int|None):
        self.blocks = []
        self.params = []
        self.grad_params = []
        self.output_dim = output_dim
        self.first_forward = True

    def set_input_dim(self, input_dim:int):
        raise NotImplementedError("set_input_dim method not implemented.")


    def __call__(self, input: np.ndarray) -> np.ndarray:
 
        if self.first_forward:
            self.set_input_dim(input.shape[-1])
            self.first_forward = False
        
        output = self.forward(input)
        return output

    def forward(self, input: np.ndarray) -> np.ndarray:
        out = input
        for block in self.blocks:
            out = block(out)
        return out
        

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = grad_output
        grad_params = []
        params = []
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        return grad
    
    def get_params(self) -> list[np.ndarray]:
        params = []
        for block in self.blocks:
            if isinstance(block, LinearBlock):
                params.append(block.param)
        return params

    def get_grad_params(self) -> list[np.ndarray]:
        grad_params = []
        for block in self.blocks:
            if isinstance(block, LinearBlock):
                grad_params.append(block._grad_param)
        return grad_params

class Model:
    def __init__(self,layers: list[Layer] | None, loss: Loss):
        self.layers: list[Layer] = layers if layers is not None else []
        self.loss = loss

    def add(self, layer: Layer):
        self.layers.append(layer)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self,loss_grad) -> None:
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def get_params(self) -> list[np.ndarray]:
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params

    def get_grad_params(self) -> list[np.ndarray]:
        grad_params = []
        for layer in self.layers:
            grad_params.extend(layer.get_grad_params())
        return grad_params