from blocks.blocks import FunctionBlock
import numpy as np

class SigmoidBlock(FunctionBlock):
    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-input))

    def input_grad(self, grad_output: np.ndarray) -> np.ndarray:
        sigmoid_derivative = self._output * (1 - self._output)
        return grad_output * sigmoid_derivative