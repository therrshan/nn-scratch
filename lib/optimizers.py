"""
Optimization algorithms for neural networks.
"""

import numpy as np


class Optimizer:
    """Base class for all optimizers."""
    
    def update(self, layers):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, layers):
        for i, layer in enumerate(layers):
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                if i not in self.velocity:
                    self.velocity[i] = {
                        'weights': np.zeros_like(layer.weights),
                        'biases': np.zeros_like(layer.biases)
                    }

                self.velocity[i]['weights'] = self.momentum * self.velocity[i]['weights'] - self.learning_rate * layer.grad_weights
                self.velocity[i]['biases'] = self.momentum * self.velocity[i]['biases'] - self.learning_rate * layer.grad_biases

                layer.weights += self.velocity[i]['weights']
                layer.biases += self.velocity[i]['biases']

            elif hasattr(layer, 'update'):
                layer.update(self.learning_rate)


class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {} 
        self.v = {}  
        self.t = 0  
    
    def update(self, layers):
        self.t += 1
        
        for i, layer in enumerate(layers):
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                if i not in self.m:
                    self.m[i] = {
                        'weights': np.zeros_like(layer.weights),
                        'biases': np.zeros_like(layer.biases)
                    }
                    self.v[i] = {
                        'weights': np.zeros_like(layer.weights),
                        'biases': np.zeros_like(layer.biases)
                    }

                self.m[i]['weights'] = self.beta1 * self.m[i]['weights'] + (1 - self.beta1) * layer.grad_weights
                self.m[i]['biases'] = self.beta1 * self.m[i]['biases'] + (1 - self.beta1) * layer.grad_biases
    
                self.v[i]['weights'] = self.beta2 * self.v[i]['weights'] + (1 - self.beta2) * np.square(layer.grad_weights)
                self.v[i]['biases'] = self.beta2 * self.v[i]['biases'] + (1 - self.beta2) * np.square(layer.grad_biases)
  
                m_hat_weights = self.m[i]['weights'] / (1 - self.beta1 ** self.t)
                m_hat_biases = self.m[i]['biases'] / (1 - self.beta1 ** self.t)
     
                v_hat_weights = self.v[i]['weights'] / (1 - self.beta2 ** self.t)
                v_hat_biases = self.v[i]['biases'] / (1 - self.beta2 ** self.t)
                
                layer.weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
                layer.biases -= self.learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)

            elif hasattr(layer, 'update'):
                layer.update(self.learning_rate)


class RMSprop(Optimizer):
    """RMSprop optimizer."""
    
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}
    
    def update(self, layers):
        for i, layer in enumerate(layers):
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                if i not in self.cache:
                    self.cache[i] = {
                        'weights': np.zeros_like(layer.weights),
                        'biases': np.zeros_like(layer.biases)
                    }

                self.cache[i]['weights'] = self.decay_rate * self.cache[i]['weights'] + (1 - self.decay_rate) * np.square(layer.grad_weights)
                self.cache[i]['biases'] = self.decay_rate * self.cache[i]['biases'] + (1 - self.decay_rate) * np.square(layer.grad_biases)

                layer.weights -= self.learning_rate * layer.grad_weights / (np.sqrt(self.cache[i]['weights']) + self.epsilon)
                layer.biases -= self.learning_rate * layer.grad_biases / (np.sqrt(self.cache[i]['biases']) + self.epsilon)

            elif hasattr(layer, 'update'):
                layer.update(self.learning_rate)