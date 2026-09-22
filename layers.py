import numpy as np

class BaseLayer:
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

class InputLayer(BaseLayer):
    def forward(self, inputs):
        return inputs

    def backward(self, grad_output):
        return grad_output

class DenseLayer(BaseLayer):
    
    def __init__(self, input_size, output_size):
        # He init: a fixed *0.1 scale starves the signal once layers stack,
        # so scale by fan-in instead (sqrt(2/n) suits the ReLU layers this
        # feeds into; softmax output layers still train fine with it).
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, grad_output, learning_rate):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.inputs.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        return grad_input

class ActivationLayer(BaseLayer):
    def __init__(self, activation, derivative):
        self.activation = activation
        self.derivative = derivative

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation(inputs)
        return self.output

    def backward(self, grad_output):
        # Optimizer.train() passes (softmax_output - y) as grad_output, which is
        # already the simplified gradient of combined softmax + cross-entropy
        # w.r.t. the pre-softmax logits, so it must pass through unchanged here.
        # Applying the softmax Jacobian on top of it would double-differentiate.
        if self.activation.__name__ == 'softmax':
            return grad_output
        return grad_output * self.derivative(self.inputs)
