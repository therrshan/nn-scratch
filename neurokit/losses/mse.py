import numpy as np

def mse(y_pred, y_true):
    loss = np.mean((y_pred - y_true)**2)
    grad = 2 * (y_pred - y_true) / y_true.shape[0]
    return loss, grad
