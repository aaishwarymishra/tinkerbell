import numpy as np

class FunctionBlock:
    def __init__(self):
        self._input: np.ndarray|None = None
        self._output = None
        self.grad_input = None
        self.grad_output = None

    def __call__(self, input):
        self._input = input
        self._output = self.forward(input)
        return self._output

    def forward(self, input):
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, grad_output):
        self.grad_output = grad_output
        self.assert_same_shape(grad_output, self._output)
        self.grad_input = self.input_grad(grad_output)
        self.assert_same_shape(self.grad_input, self._input)
        return self.grad_input
        

    def input_grad(self, grad_output):
        raise NotImplementedError("Backward method not implemented.")

    def assert_same_shape(self, a, b):
        assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}" 


class LinearBlock(FunctionBlock):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.grad_param = None

    def backward(self, grad_output):
        self.assert_same_shape(grad_output, self._output)
        self.grad_input = self.input_grad(grad_output)
        self.assert_same_shape(self.grad_input, self._input)
        self.grad_param = self.param_grad(grad_output)
        self.assert_same_shape(self.grad_param, self.param)
        return self.grad_input

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
        if self._input is None:
            raise ValueError("Input not set; build model by sending input first.")
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
