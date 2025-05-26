import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, d_out, lr):
        dW = self.X.T @ d_out
        db = np.sum(d_out, axis=0, keepdims=True)
        self.W -= lr * dW
        self.b -= lr * db
        return d_out @ self.W.T
