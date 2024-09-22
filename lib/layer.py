import numpy as np
from math_utils import sigmoid, sigmoid_derivative, relu, relu_derivative, softmax, softmax_derivative


class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, input_data):
        if input_data.shape[0] != 1:
            self.input = input_data.reshape((input_data.shape[0], 1))
        else:
            self.input = input_data.reshape((input_data.shape[1], 1))
        self.z = np.dot(self.input.T, self.weights) + self.bias

        if self.activation == 'sigmoid':
            self.output = sigmoid(self.z)
        elif self.activation == 'relu':
            self.output = relu(self.z)
        elif self.activation == 'softmax':
            self.output = softmax(self.z)
        else:
            raise ValueError("Unsupported activation function")

        return self.output

    def backward(self, output_error, learning_rate):
        if self.activation == 'sigmoid':
            activation_derivative = sigmoid_derivative(self.output)
        elif self.activation == 'relu':
            activation_derivative = relu_derivative(self.z)
        elif self.activation == 'softmax':
            activation_derivative = softmax_derivative(self.z)
        else:
            raise ValueError("Unsupported activation function")

        output_error = output_error * activation_derivative
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * np.sum(output_error, axis=0, keepdims=True)

        return input_error