"""
Loss functions for neural networks.
"""

import numpy as np

class Loss:
    """Base class for all loss functions."""
    
    def forward(self, predictions, targets):
        raise NotImplementedError
    
    def backward(self, predictions, targets):
        raise NotImplementedError


class MSE(Loss):
    """Mean Squared Error loss function."""
    
    def forward(self, predictions, targets):
        return np.mean(np.square(predictions - targets))
    
    def backward(self, predictions, targets):
        batch_size = predictions.shape[0]
        return 2 * (predictions - targets) / batch_size


class CrossEntropy(Loss):
    """Cross-Entropy loss function for classification."""
    
    def forward(self, predictions, targets):
        epsilon = 1e-12
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(np.sum(targets * np.log(predictions), axis=1))
    
    def backward(self, predictions, targets):
        batch_size = predictions.shape[0]
        epsilon = 1e-12
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -(targets / predictions) / batch_size


class BinaryCrossEntropy(Loss):
    """Binary Cross-Entropy loss function for classification."""
    
    def forward(self, predictions, targets):
        epsilon = 1e-12
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    def backward(self, predictions, targets):
        batch_size = predictions.shape[0]
        epsilon = 1e-12
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -(targets / predictions - (1 - targets) / (1 - predictions)) / batch_size