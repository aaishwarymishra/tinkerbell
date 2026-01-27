# Tinkerbell

Tinkerbell is a lightweight, educational deep learning framework built entirely from scratch using **NumPy**. It implements automatic differentiation with both forward-mode and reverse-mode autograd, modular layer architecture, optimizers, and loss functions to demonstrate the core concepts of neural networks.

## Features

- **Automatic Differentiation**:
  - `Tensor`: Reverse-mode autograd for efficient gradient computation.
  - `ForwardTensor`: Forward-mode autograd for educational purposes.
- **Modular Layers**:
  - `DenseLayer`: Fully connected layer.
  - `SigmoidLayer`: Sigmoid activation layer.
  - `SeqLayer`: Sequential container for stacking layers.
- **Optimizers**:
  - `SGD`: Stochastic Gradient Descent.
- **Loss Functions**:
  - `MSELoss`: Mean Squared Error.
- **Additional Tools**:
  - `SimpleLinearRegression`: Simplified linear regression implementation.
  - Progress tracking with `tqdm`.
  - Visualization with `matplotlib`.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   pip install tinkerbell
   ```



## Quick Start

### Using Tensors for Autograd

```python
import numpy as np
from tinkerbell.tensor import Tensor

# Create tensors with gradient tracking
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = Tensor([4.0, 5.0, 6.0], requires_grad=True)

# Perform operations
z = x * y + Tensor([1.0, 1.0, 1.0])
z.grad = np.ones_like(z.data)  # Set gradient of z to ones
z.backward()  # Compute gradients

print(x.grad)  # Gradients w.r.t. x
print(y.grad)  # Gradients w.r.t. y
```

### Building and Training a Neural Network

```python
import numpy as np
from tinkerbell.tensor import Tensor
from tinkerbell.layers import DenseLayer, SigmoidLayer, SeqLayer
from tinkerbell.loss import MSELoss
from tinkerbell.optimizer import SGD

# Create dummy data
x = np.random.random((100, 3))
y = np.dot(x, np.array([[2.0], [3.0], [4.0]])) + 5.0

# Build the model
model = SeqLayer([
    DenseLayer(input_dim=3, output_dim=4),
    SigmoidLayer(),
    DenseLayer(input_dim=4, output_dim=1)
])

# Set up loss and optimizer
loss_fn = MSELoss()
optimizer = SGD(learning_rate=0.01)
model.trainer(loss=loss_fn, optimizer=optimizer)

# Train the model
model.fit(Tensor(x), Tensor(y), epochs=100, batch_size=10)

# Make predictions
predictions = model(Tensor(x))
```

### Simple Linear Regression

```python
from tinkerbell.simple_linear_regression import SimpleLinearRegression
import numpy as np

# Create data
x = np.random.random((100, 2))
y = np.dot(x, np.array([[2.0], [3.0]])) + 1.0

# Train model
model = SimpleLinearRegression()
model.fit(x, y, learning_rate=0.01, epochs=1000)

# Predict
predictions = model.forward(x)
```

## Project Structure

```
src/
└── tinkerbell/
    ├── __init__.py
    ├── tensor.py          # Tensor classes (Tensor, ForwardTensor, Parameter)
    ├── functions.py       # Activation functions (Sigmoid)
    ├── layers.py          # Layer implementations (DenseLayer, SigmoidLayer, SeqLayer)
    ├── loss.py            # Loss functions (MSELoss)
    ├── optimizer.py       # Optimization algorithms (SGD)
    ├── simple_linear_regression.py  # Simplified linear regression
    └── legacy/            # Older implementations
        ├── __init__.py
        ├── core.py
        ├── functions.py
        ├── layers.py
        ├── loss.py
        ├── models.py
        └── optimizer.py
main.py                    # Entry point
pyproject.toml             # Project configuration
test.ipynb                 # Jupyter notebook for testing and examples
README.md                  # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Authors

- **Aaiswarya Mishra** - [aaishwarymishra@gmail.com](mailto:aaishwarymishra@gmail.com)
