import numpy as np
from lib.network import NeuralNetwork
from lib.layer import Layer
from keras.datasets import mnist
from keras.utils import to_categorical


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize the input data
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test


def main():
    # Load MNIST data
    X_train, y_train, X_test, y_test = load_mnist_data()

    # Create neural network
    nn = NeuralNetwork()
    nn.add(Layer(784, 512, 'relu')) # Input layer
    nn.add(Layer(512, 256, 'relu')) # Hidden layer
    nn.add(Layer(256, 128, 'relu')) # Hidden layer
    nn.add(Layer(128, 64, 'relu'))  # Hidden layer
    nn.add(Layer(64, 10, 'softmax'))  # Output layer

    # Set loss function
    nn.set_loss(mse, mse_derivative)

    # Train the network
    nn.fit(X_train, y_train, epochs=10, learning_rate=0.1)

    # Evaluate the model
    correct = 0
    total = len(X_test)

    for i in range(total):
        prediction = nn.predict(X_test[i])
        if np.argmax(prediction) == np.argmax(y_test[i]):
            correct += 1

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()