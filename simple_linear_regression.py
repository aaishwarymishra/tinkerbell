import numpy as np
from tqdm.auto import tqdm

class SimpleLinearRegression:
    def __init__(self):
        self.weight: np.ndarray = None
        self.bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert self.weight is not None and self.bias is not None, "Model is not trained yet."
        assert x.shape[1] == self.weight.shape[0], "Input features do not match weight dimensions."
        return np.dot(x, self.weight) + self.bias

    def mse_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    def backward(self,x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,loss) -> tuple[np.ndarray, float]:
        loss = self.mse_loss(y_true, y_pred)
        n_samples = x.shape[0]

        dloss_dpred = -2 * (y_true - y_pred) / n_samples
        dpred_dN = np.ones_like(y_true)
        dN_dweight = np.transpose(x)
        dpred_dbias = np.ones_like(y_true)

        weights_grad = np.matmul(dN_dweight, (dloss_dpred * dpred_dN))
        bias_grad = np.sum(dloss_dpred * dpred_dbias)
        return weights_grad, bias_grad

    def fit(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, epochs: int = 1000,loss='mse'):
        assert x.shape[0] == y.shape[0], "Number of samples in x and y must be the same."
        n_samples, n_features = x.shape
        self.weight = np.random.rand(n_features, 1)
        self.bias = np.random.rand(1,)
        if loss == 'mse':
            loss_function = self.mse_loss

        for epoch in tqdm(range(epochs)):
            y_pred = self.forward(x)
            w_grad, b_grad = self.backward(x, y, y_pred,loss=loss_function)

            self.weight -= learning_rate * w_grad
            self.bias -= learning_rate * b_grad
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {error:.4f}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.weight) + self.bias

if __name__ == "__main__":
    # Example usage
    x = np.random.random((100, 3))
    y = np.matmul(x, np.array([[2.0], [3.0], [4.0]])) + 5.0 

    model = SimpleLinearRegression()
    model.fit(x, y, learning_rate=0.01, epochs=1000)

    print("Trained weights:", model.weight)
    print("Trained bias:", model.bias)