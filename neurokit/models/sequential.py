class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, d_out, lr):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out, lr)

    def train(self, X, y, loss_fn, epochs, lr):
        for epoch in range(epochs):
            preds = self.forward(X)
            loss, d_loss = loss_fn(preds, y)
            self.backward(d_loss, lr)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
