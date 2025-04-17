"""
Layer implementations for neural networks.
"""

import numpy as np


class Layer:
    
    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError
    
    def update(self, learning_rate):
        pass 


class Dense(Layer):
    """Fully connected layer."""
    
    def __init__(self, input_size, output_size):

        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros(output_size)
        
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)
    
    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases
    
    def backward(self, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        self.grad_weights = np.dot(self.inputs.T, grad_output)
        self.grad_biases = np.sum(grad_output, axis=0)
        
        return grad_input
    
    def update(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases


class Dropout(Layer):
    """Dropout layer for regularization."""
    
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, inputs, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape) / (1 - self.dropout_rate)
            return inputs * self.mask
        else:
            return inputs
    
    def backward(self, grad_output):
        return grad_output * self.mask


class BatchNormalization(Layer):
    """Batch Normalization layer."""
    
    def __init__(self, input_size, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        
        self.gamma = np.ones(input_size)
        self.beta = np.zeros(input_size)
        
        self.running_mean = np.zeros(input_size)
        self.running_var = np.ones(input_size)
        
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)
    
    def forward(self, inputs, training=True):
        if training:
            self.inputs = inputs
            self.batch_size = inputs.shape[0]

            self.batch_mean = np.mean(inputs, axis=0)
            self.batch_var = np.var(inputs, axis=0)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            self.x_norm = (inputs - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)

            return self.gamma * self.x_norm + self.beta

        else:
            x_norm = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.gamma * x_norm + self.beta
    
    def backward(self, grad_output):
        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=0)
        self.grad_beta = np.sum(grad_output, axis=0)

        dx_norm = grad_output * self.gamma
        dvar = np.sum(dx_norm * (self.inputs - self.batch_mean) * -0.5 * np.power(self.batch_var + self.epsilon, -1.5), axis=0)
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.batch_var + self.epsilon), axis=0) + dvar * np.mean(-2 * (self.inputs - self.batch_mean), axis=0)
        
        grad_input = dx_norm / np.sqrt(self.batch_var + self.epsilon) + dvar * 2 * (self.inputs - self.batch_mean) / self.batch_size + dmean / self.batch_size
        
        return grad_input
    
    def update(self, learning_rate):
        self.gamma -= learning_rate * self.grad_gamma
        self.beta -= learning_rate * self.grad_beta


class Flatten(Layer):
    """Flatten layer to convert multi-dimensional inputs to 1D."""
    
    def forward(self, inputs):
        self.input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)
    
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)