from blocks.blocks import Layer, WeightMatrixBlock, BiasBlock
from blocks.functions import SigmoidBlock

import numpy as np

class DenseLayer(Layer):
    def __init__(self, output_dim: int,seed: int | None = None):
        super().__init__(output_dim)
        self.seed = seed

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