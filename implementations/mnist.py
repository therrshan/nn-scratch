"""
Example implementation of MNIST digit classification using our neural network.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Import our neural network components
from lib.network import NeuralNetwork
from lib.layers import Dense, Dropout, BatchNormalization, Flatten
from lib.activations import ReLU, Softmax
from lib.losses import CrossEntropy
from lib.optimizers import Adam
from lib.utils import one_hot_encode, accuracy


def load_mnist():
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1)
    
    X = mnist['data'].to_numpy().astype(np.float32)
    y = mnist['target'].to_numpy().astype(np.int32)
    
    X = X / 255.0
    y_one_hot = one_hot_encode(y, 10)

    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_test, y_test


def create_model():
    """Create and compile the neural network."""
    model = NeuralNetwork()
    
    model.add(Dense(784, 512))
    model.add(ReLU())
    #model.add(BatchNormalization(512))
    model.add(Dropout(0.2))
    
    model.add(Dense(512, 256))
    model.add(ReLU())
    #model.add(BatchNormalization(256))
    model.add(Dropout(0.2))
    
    model.add(Dense(256, 10))
    model.add(Softmax())
    
    model.compile(
        loss=CrossEntropy(),
        optimizer=Adam(learning_rate=0.0001)
    )
    
    return model


def plot_history(history):
    """Plot training and validation metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('mnist_training_history.png')
    plt.show()


def plot_predictions(model, X_test, y_test, num_samples=5):
    predictions = model.predict(X_test[:num_samples])
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test[:num_samples], axis=1)

    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    
    for i in range(num_samples):
        axes[i].imshow(X_test[i].reshape(28, 28), cmap='gray')
        
        title = f"True: {true_labels[i]}\nPred: {pred_labels[i]}"
        axes[i].set_title(title)

        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png')
    plt.show()


def main():
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    model = create_model()

    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=20,
        validation_data=(X_test, y_test),
        early_stopping=True,
        patience=10,
        lambda_reg=0.0001,
        verbose=2
    )

    test_loss, test_pred = model.evaluate(X_test, y_test)
    test_acc = accuracy(test_pred, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    plot_history(history)
    plot_predictions(model, X_test, y_test)
    model.save_weights('mnist_model.npy')
    print("Model weights saved to 'mnist_model.npy'")


if __name__ == "__main__":
    main()