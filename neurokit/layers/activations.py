import numpy as np

class ReLU:
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask

    def backward(self, d_out, lr):
        return d_out * self.mask
