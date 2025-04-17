"""
Activation functions for neural networks.
"""

import numpy as np


class Activation:
    
    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError


class ReLU(Activation):
    """Relu activation function."""

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)
    
    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.inputs <= 0] = 0
        return grad_input


class Sigmoid(Activation):
    """Sigmoid activation function."""

    def forward(self, inputs):
        self.outputs = 1.0 / (1.0 + np.exp(-inputs))
        return self.outputs
    
    def backward(self, grad_output):
        return grad_output * self.outputs * (1 - self.outputs)


class Softmax(Activation):
    """Softmax activation function."""
    
    def forward(self, inputs):
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(shifted_inputs)
        self.outputs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.outputs
    
    def backward(self, grad_output):
        return grad_output


class Tanh(Activation):
    """Hyperbolic Tangent activation function."""
    
    def forward(self, inputs):
        self.outputs = np.tanh(inputs)
        return self.outputs
    
    def backward(self, grad_output):
        return grad_output * (1 - np.square(self.outputs))


class LeakyReLU(Activation):
    """Leaky ReLU activation function."""
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, inputs):
        self.inputs = inputs
        return np.where(inputs > 0, inputs, inputs * self.alpha)
    
    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.inputs <= 0] = grad_input[self.inputs <= 0] * self.alpha
        return grad_input