"""Train a small MLP on a synthetic 3-arm spiral and report accuracy.

Usage:
    python demo.py
"""
import numpy as np

from activations import relu, relu_derivative, softmax
from layers import ActivationLayer, DenseLayer
from models import BaseANN
from optimizers import Optimizer


def make_spiral(points_per_class=100, classes=3, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((points_per_class * classes, 2))
    y = np.zeros((points_per_class * classes, classes))
    for class_idx in range(classes):
        idx = slice(points_per_class * class_idx, points_per_class * (class_idx + 1))
        r = np.linspace(0.0, 1.0, points_per_class)
        t = (
            class_idx * 4
            + np.linspace(0.0, 4.0, points_per_class)
            + rng.normal(scale=noise, size=points_per_class)
        )
        X[idx] = np.column_stack([r * np.sin(t * 2.5), r * np.cos(t * 2.5)])
        y[idx, class_idx] = 1.0
    return X, y


def build_model(hidden_size=16):
    model = BaseANN()
    sizes = [2, hidden_size, hidden_size, 3]
    for i in range(len(sizes) - 1):
        model.add(DenseLayer(sizes[i], sizes[i + 1]))
        is_output = i == len(sizes) - 2
        model.add(ActivationLayer(softmax, None) if is_output else ActivationLayer(relu, relu_derivative))
    return model


def main():
    X, y = make_spiral()
    model = build_model()
    optimizer = Optimizer(model, learning_rate=0.01, clip_norm=5.0)
    optimizer.train(X, y, epochs=250, batch_size=32)

    predictions = model.predict(X)
    accuracy = np.mean(np.argmax(y, axis=1) == np.argmax(predictions, axis=1))
    print(f"Final training accuracy: {accuracy * 100:.1f}%")


if __name__ == "__main__":
    main()
