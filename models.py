import numpy as np

class BaseANN:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        """
        Forward pass through all layers.
        :param X: Input data.
        :return: Network output.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output, learning_rate, clip_value=1.0):
        """
        Backward pass through all layers in reverse order with gradient clipping.
        :param grad_output: Initial gradient.
        :param learning_rate: Learning rate for the updates.
        :param clip_value: Maximum allowed norm of the gradient.
        :return: Final gradient after backpropagation.
        """
        for layer in reversed(self.layers):
            if isinstance(layer, DenseLayer):
                # Compute gradient for layer
                grad_output = layer.backward(grad_output, learning_rate)
                # Clip the gradients to avoid explosion
                grad_norm = np.linalg.norm(grad_output)
                if grad_norm > clip_value:
                    grad_output = (grad_output / grad_norm) * clip_value
            else:
                grad_output = layer.backward(grad_output)
        return grad_output

    def predict(self, X):
        return self.forward(X)
