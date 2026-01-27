from tinkerbell.legacy.core import  WeightMatrixBlock, BiasBlock, LinearBlock
from tinkerbell.legacy.functions import SigmoidBlock
import numpy as np

class Layer:
    def __init__(self,output_dim:int|None,seed: int | None = None):
        self.blocks = []
        self.output_dim = output_dim
        self.grad_input = None
        self.grad_output = None
        self.first_forward = True
        self._input = None
        self._output = None
        self.seed = seed

    def set_input_dim(self, input_dim:int):
        raise NotImplementedError("set_input_dim method not implemented.")


    def __call__(self, input: np.ndarray) -> np.ndarray:
 
        if self.first_forward:
            self.set_input_dim(input.shape[-1])
            self.first_forward = False
        self._input = input
        output = self.forward(input)
        self._output = output
        return output

    def forward(self, input: np.ndarray) -> np.ndarray:
        out = input
        for block in self.blocks:
            out = block(out)
        return out
        

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.grad_output = grad_output
        grad = grad_output
        for block in reversed(self.blocks):
            grad = block.backward(grad) 
        self.grad_input = grad
        return grad
    
    def get_params(self):
        params = []
        for block in self.blocks:
            if isinstance(block, LinearBlock):
                if block.param is None:
                    raise ValueError("Parameter not set; build model by sending input first.")
                yield block.param


    def get_grad_params(self):
        for block in self.blocks:
            if isinstance(block, LinearBlock):
                if block.grad_param is None:
                    raise ValueError("Gradient parameter not set; build model by backward pass first.")
                yield block.grad_param


class DenseLayer(Layer):
    def __init__(self, output_dim: int|None,seed: int | None = None):
        super().__init__(output_dim,seed=seed)


    def set_input_dim(self, input_dim: int):
        if self.seed is not None:
            np.random.seed(self.seed)
        out_dim = self.output_dim if self.output_dim is not None else 1
        weight = np.random.randn(input_dim, out_dim) 
        bias = np.random.randn(1, out_dim) 
        self.blocks.append(WeightMatrixBlock(weight))
        self.blocks.append(BiasBlock(bias))
        self.params = self.get_params()

class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__(output_dim=None)
        self.blocks.append(SigmoidBlock())
    
    def set_input_dim(self, input_dim: int):
        pass