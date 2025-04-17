"""
Utility functions for neural networks.
"""

import numpy as np


def one_hot_encode(labels, num_classes):
    batch_size = labels.shape[0]
    one_hot = np.zeros((batch_size, num_classes))
    one_hot[np.arange(batch_size), labels] = 1
    return one_hot


def accuracy(predictions, targets):
    pred_labels = np.argmax(predictions, axis=1)
    
    if len(targets.shape) > 1 and targets.shape[1] > 1:
        true_labels = np.argmax(targets, axis=1)
    else:
        true_labels = targets
    
    return np.mean(pred_labels == true_labels)


def shuffle_data(inputs, targets):
    assert len(inputs) == len(targets), "Inputs and targets must have the same number of samples"   
    indices = np.random.permutation(len(inputs))
    return inputs[indices], targets[indices]


def train_val_split(inputs, targets, val_ratio=0.2):
    assert 0 <= val_ratio < 1, "Validation ratio must be between 0 and 1"
    assert len(inputs) == len(targets), "Inputs and targets must have the same number of samples"

    inputs, targets = shuffle_data(inputs, targets)
    split_idx = int(len(inputs) * (1 - val_ratio))

    train_inputs, val_inputs = inputs[:split_idx], inputs[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]
    
    return train_inputs, train_targets, val_inputs, val_targets


def batch_iterator(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets), "Inputs and targets must have the same number of samples"
    
    if shuffle:
        inputs, targets = shuffle_data(inputs, targets)
    
    num_samples = len(inputs)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        yield inputs[start_idx:end_idx], targets[start_idx:end_idx]


def standardize(data, mean=None, std=None):
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
        std = np.where(std == 0, 1, std)
    
    return (data - mean) / std, mean, std


def l2_regularization(layers, lambda_reg):
    reg_loss = 0.0
    
    for layer in layers:
        if hasattr(layer, 'weights'):
            reg_loss += np.sum(np.square(layer.weights))
            if hasattr(layer, 'grad_weights'):
                layer.grad_weights += 2 * lambda_reg * layer.weights
    
    return lambda_reg * reg_loss