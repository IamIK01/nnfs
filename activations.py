import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def softmax_derivative(softmax_output):
    # Extract the batch size and number of classes
    batch_size = softmax_output.shape[0]
    num_classes = softmax_output.shape[1]
    
    # Create an empty array to store Jacobian matrices for each sample
    jacobian_matrices = np.zeros((batch_size, num_classes, num_classes))
    
    # Compute the Jacobian matrix for each sample
    for i in range(batch_size):
        s = softmax_output[i]
        # Compute the diagonal terms
        diagonal = np.diag(s * (1 - s))
        # Compute the off-diagonal terms using outer product
        outer = np.outer(s, s)
        # The Jacobian matrix is diagonal - outer
        jacobian_matrices[i] = diagonal - outer

    return jacobian_matrices

# Sigmoid activation function
def sigmoid(x):
    """
    Sigmoid activation function.
    :param x: Input data.
    :return: Sigmoid-activated values.
    """
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    """
    Derivative of the sigmoid function.
    :param x: Sigmoid-activated values.
    :return: Gradient of the sigmoid function.
    """
    return x * (1 - x)

# Tanh activation function
def tanh(x):
    """
    Tanh activation function.
    :param x: Input data.
    :return: Tanh-activated values.
    """
    return np.tanh(x)

# Derivative of tanh
def tanh_derivative(x):
    """
    Derivative of the tanh function.
    :param x: Tanh-activated values.
    :return: Gradient of the tanh function.
    """
    return 1 - np.power(x, 2)

def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation function.
    :param x: Input data.
    :param alpha: Slope for negative values (default is 0.01).
    :return: Activated values.
    """
    return np.where(x > 0, x, x * alpha)

def leaky_relu_derivative(x, alpha=0.01):
    """
    Derivative of the Leaky ReLU function.
    :param x: Input data.
    :param alpha: Slope for negative values (default is 0.01).
    :return: Gradient of the Leaky ReLU function.
    """
    return np.where(x > 0, 1, alpha)
