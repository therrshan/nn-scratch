"""
Neural Network implementation combining layers, activations, and optimizers.
"""

import numpy as np
from copy import deepcopy


class NeuralNetwork:
    """Neural Network class."""
    
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.best_val_loss = float('inf')
        self.best_model = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def add(self, layer):
        self.layers.append(layer)
    
    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
    
    def forward(self, inputs, training=True):
        x = inputs
        for layer in self.layers:
            if hasattr(layer, 'training') or isinstance(layer, type) and hasattr(layer, 'training'):
                x = layer.forward(x, training)
            else:
                x = layer.forward(x)
        
        return x
    
    def backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        return grad
    
    def train_step(self, inputs, targets):
        predictions = self.forward(inputs, training=True)
        loss_value = self.loss.forward(predictions, targets)
        grad_output = self.loss.backward(predictions, targets)
        self.backward(grad_output)
        self.optimizer.update(self.layers)
        
        return loss_value, predictions
    
    def evaluate(self, inputs, targets):
        predictions = self.forward(inputs, training=False)
        loss_value = self.loss.forward(predictions, targets)   
        return loss_value, predictions
    
    def fit(self, x_train, y_train, batch_size=32, epochs=10, 
            validation_data=None, early_stopping=False, patience=5, 
            lambda_reg=0.0, verbose=1):
        """
        Train the network.
        
        Args:
            x_train: Training inputs
            y_train: Training targets
            batch_size: Size of each mini-batch (default: 32)
            epochs: Number of epochs to train (default: 10)
            validation_data: Tuple of (validation inputs, validation targets) (default: None)
            early_stopping: Whether to use early stopping (default: False)
            patience: Number of epochs with no improvement to wait before stopping (default: 5)
            lambda_reg: L2 regularization strength (default: 0.0)
            verbose: Verbosity level (0: silent, 1: progress bar, 2: one line per epoch) (default: 1)
            
        Returns:
            Training history
        """
        num_samples = len(x_train)
        best_val_loss = float('inf')
        patience_counter = 0
        
        val_inputs, val_targets = None, None
        if validation_data is not None:
            val_inputs, val_targets = validation_data
        
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]

            epoch_loss = 0.0
            epoch_acc = 0.0
            
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_inputs = x_train_shuffled[i:end_idx]
                batch_targets = y_train_shuffled[i:end_idx]

                batch_loss, batch_predictions = self.train_step(batch_inputs, batch_targets)

                if lambda_reg > 0:
                    from .utils import l2_regularization
                    reg_loss = l2_regularization(self.layers, lambda_reg)
                    batch_loss += reg_loss

                try:
                    from .utils import accuracy
                    batch_acc = accuracy(batch_predictions, batch_targets)
                    epoch_acc += batch_acc * len(batch_inputs) / num_samples
                except:
                    epoch_acc = None

                epoch_loss += batch_loss * len(batch_inputs) / num_samples

                if verbose == 1:
                    progress = (i + len(batch_inputs)) / num_samples
                    progress_bar = '#' * int(progress * 20) + '-' * (20 - int(progress * 20))
                    if epoch_acc is not None:
                        print(f"\rEpoch {epoch+1}/{epochs} [{progress_bar}] - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}", end='')
                    else:
                        print(f"\rEpoch {epoch+1}/{epochs} [{progress_bar}] - loss: {epoch_loss:.4f}", end='')

            val_loss, val_acc = None, None
            if val_inputs is not None and val_targets is not None:
                val_loss, val_predictions = self.evaluate(val_inputs, val_targets)

                if lambda_reg > 0:
                    from .utils import l2_regularization
                    reg_loss = l2_regularization(self.layers, lambda_reg)
                    val_loss += reg_loss

                try:
                    from .utils import accuracy
                    val_acc = accuracy(val_predictions, val_targets)
                except:
                    val_acc = None

            if verbose > 0:
                if val_loss is not None and val_acc is not None:
                    print(f"\rEpoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
                elif val_loss is not None:
                    print(f"\rEpoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
                elif epoch_acc is not None:
                    print(f"\rEpoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")
                else:
                    print(f"\rEpoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")

            self.training_history['train_loss'].append(epoch_loss)
            if epoch_acc is not None:
                self.training_history['train_acc'].append(epoch_acc)
            if val_loss is not None:
                self.training_history['val_loss'].append(val_loss)
            if val_acc is not None:
                self.training_history['val_acc'].append(val_acc)

            if early_stopping and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_val_loss = val_loss
                    patience_counter = 0

                    self.best_model = deepcopy(self.layers)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose > 0:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
 
        if early_stopping and self.best_model is not None:
            self.layers = self.best_model
        
        return self.training_history
    
    def predict(self, inputs):
        return self.forward(inputs, training=False)
    
    def save_weights(self, filename):
        weights = []
        
        for layer in self.layers:
            layer_weights = {}
            
            if hasattr(layer, 'weights'):
                layer_weights['weights'] = layer.weights
            
            if hasattr(layer, 'biases'):
                layer_weights['biases'] = layer.biases
            
            if hasattr(layer, 'gamma'):
                layer_weights['gamma'] = layer.gamma
            
            if hasattr(layer, 'beta'):
                layer_weights['beta'] = layer.beta
            
            if hasattr(layer, 'running_mean'):
                layer_weights['running_mean'] = layer.running_mean
            
            if hasattr(layer, 'running_var'):
                layer_weights['running_var'] = layer.running_var
            
            weights.append(layer_weights)
        
        np.save(filename, weights)
    
    def load_weights(self, filename):
        weights = np.load(filename, allow_pickle=True)
        
        assert len(weights) == len(self.layers), "Number of layers in the saved weights does not match the network"
        
        for i, layer in enumerate(self.layers):
            layer_weights = weights[i]
            
            if hasattr(layer, 'weights') and 'weights' in layer_weights:
                layer.weights = layer_weights['weights']
            
            if hasattr(layer, 'biases') and 'biases' in layer_weights:
                layer.biases = layer_weights['biases']
            
            if hasattr(layer, 'gamma') and 'gamma' in layer_weights:
                layer.gamma = layer_weights['gamma']
            
            if hasattr(layer, 'beta') and 'beta' in layer_weights:
                layer.beta = layer_weights['beta']
            
            if hasattr(layer, 'running_mean') and 'running_mean' in layer_weights:
                layer.running_mean = layer_weights['running_mean']
            
            if hasattr(layer, 'running_var') and 'running_var' in layer_weights:
                layer.running_var = layer_weights['running_var']