# Tinkerbell 

Tinkerbell is a lightweight, educational deep learning framework built entirely from scratch using **NumPy**. It is designed to demonstrate the fundamental building blocks of neural networks, including forward and backward propagation, gradient descent, and modular layer architecture.

## Features

- **Modular Architecture**: Built with `FunctionBlock` and `Layer` abstractions.
- **Automated Backward Pass**: Manual implementation of gradients for all blocks.
- **Layers**:
  - `DenseLayer`: Standard fully connected layer.
  - `SigmoidLayer`: Sigmoid activation function.
- **Optimizers**:
  - `SGD`: Stochastic Gradient Descent.
- **Loss Functions**:
  - `MSELoss`: Mean Squared Error.
- **Additional Tools**:
  - `SimpleLinearRegression`: A simplified implementation for linear problems.
  - Progress tracking with `tqdm`.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tinkerbell.git
   cd tinkerbell
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

## Quick Start

Building and training a model in Tinkerbell is straightforward:

```python
import numpy as np
from tinkerbell.models import Model
from tinkerbell.layers import DenseLayer, SigmoidLayer
from tinkerbell.loss import MSELoss
from tinkerbell.optimizer import SGD

# 1. Create dummy data
x = np.random.random((100, 3))
y = np.dot(x, np.array([[2.0], [3.0], [4.0]])) + 5.0 

# 2. Initialize the Model
model = Model(layers=[
    DenseLayer(output_dim=4),
    SigmoidLayer(),
    DenseLayer(output_dim=1)
])

# 3. Setup Optimizer and Loss
optimizer = SGD(model.get_params(), learning_rate=0.01)
model.trainer(loss=MSELoss(), optimizer=optimizer)

# 4. Fit the Model
model.fit(x, y, epochs=100, batch_size=10)

# 5. Make Predictions
predictions = model(x)
```

## Project Structure

```text
src/
└── tinkerbell/
    ├── core.py        # Base classes for blocks (WeightMatrix, Bias)
    ├── functions.py   # Activation functions (Sigmoid)
    ├── layers.py      # Layer implementations (Dense, Sigmoid)
    ├── loss.py        # Loss functions (MSE)
    ├── models.py      # Model class and training logic
    └── optimizer.py   # Optimization algorithms (SGD)
```

## Authors

- **Aaiswarya Mishra** - [aaishwarymishra@gmail.com](mailto:aaishwarymishra@gmail.com)
