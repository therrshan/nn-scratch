# Neural Networks from Scratch

This project is an implementation of neural networks built from the ground up, without using deep learning frameworks like PyTorch or TensorFlow. The aim is to understand the mathematical foundations and underlying mechanisms of neural networks by manually implementing key components such as layers, activation functions, and backpropagation.
 
## TODO List

- [ ] Implement standard activations (ReLU, Sigmoid, Softmax)
- [ ] Forward & Backward Propagation for layers
- [ ] Implement early stopping and dropout
- [ ] Implement advanced activations for more complex implementations
- [ ] Implement custom loss functions (e.g., Cross Entropy)
- [ ] Build custom optimizers (e.g., SGD, Adam)
- [ ] Add support for batch normalization and regularization
  
As the project progresses, additional features and more complex network architectures will be added. If you want to contribute or have any suggestions, feel free to reach out!

## Project Structure

```bash
nn-scratch/
│
├── implementations/
│   └── mnist.py          # Example implementation of MNIST digit classification
│
├── lib/
│   ├── layer.py          # Contains layer definitions and forward/backward propagation
│   ├── math_utils.py     # Mathematical utilities for neural network operations
│   └── network.py        # Network class combining layers, activations, and propagation
└── 
