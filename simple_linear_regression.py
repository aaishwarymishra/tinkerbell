from matplotlib.pylab import permutation
from fontTools.afmLib import error
import math
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

    def backward(self,x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,loss="mse") -> tuple[np.ndarray, float]:
        n_samples = x.shape[0]

        if loss == 'mse':
            dloss_dpred = -2 * (y_true - y_pred) / n_samples
        elif loss == 'rmse':
            rmse = self.rmse_loss(y_true, y_pred) + 1e-8  # to avoid division by zero
            dloss_dpred = - (y_true - y_pred) / (rmse * n_samples)
        dpred_dM = np.ones_like(y_true)
        dM_dweight = np.transpose(x)
        dpred_dbias = np.ones_like(y_true)

        weights_grad = np.matmul(dM_dweight, (dloss_dpred * dpred_dM))
        bias_grad = np.sum(dloss_dpred * dpred_dbias)
        return weights_grad, bias_grad

    def fit(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, epochs: int = 1000,loss='mse', batch_size: int = 32):
        assert x.shape[0] == y.shape[0], "Number of samples in x and y must be the same."
        n_samples, n_features = x.shape
        self.weight = np.random.rand(n_features, 1)
        self.bias = np.random.rand(1,)
        if loss == 'mse':
            loss_function = self.mse_loss
        elif loss == 'rmse':
            loss_function = self.rmse_loss   

        batches = math.ceil(n_samples / batch_size)

        for epoch in tqdm(range(epochs)):
            avg_loss = 0
            permutation = np.random.permutation(n_samples)
            for x_batch, y_batch in self.get_batch(x, y, batch_size,permutation):
                y_pred = self.forward(x_batch)
                w_grad, b_grad = self.backward(x_batch, y_batch, y_pred,loss=loss)
                avg_loss += loss_function(y_batch, y_pred)
                self.weight -= learning_rate * w_grad
                self.bias -= learning_rate * b_grad
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {(avg_loss/batches):.4f}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.weight) + self.bias

    def get_batch(self, x: np.ndarray, y: np.ndarray, batch_size: int,permutation:list[int]):
        n_samples = x.shape[0]
        for i in range(0, n_samples, batch_size):
            x_batch = x[permutation[i:i + batch_size]]
            y_batch = y[permutation[i:i + batch_size]]
            yield x_batch, y_batch
    
    def rmse_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

if __name__ == "__main__":
    # Example usage
    x = np.random.random((100, 3))
    y = np.matmul(x, np.array([[2.0], [3.0], [4.0]])) + 5.0 

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

    model = SimpleLinearRegression()
    model.fit(x, y, learning_rate=0.01, epochs=1000,loss="rmse")

    print("Trained weights:", model.weight)
    print("Trained bias:", model.bias)
    print("R-squared:", model.r_squared(y, model.predict(x)))
    print("RMSE:", model.rmse_loss(y, model.predict(x)))
    print("MSE:", model.mse_loss(y, model.predict(x)))
