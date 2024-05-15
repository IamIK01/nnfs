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
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.random.randn(1, output_size) * 0.1

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
        if self.activation.__name__ == 'softmax':
            return np.einsum('ijk,ik->ij', softmax_derivative(self.output), grad_output)
        return grad_output * self.derivative(self.inputs)
