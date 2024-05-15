import numpy as np
from tqdm import tqdm

class Optimizer:
    def __init__(self, model, learning_rate=0.01, clip_norm=1.0):
        self.model = model
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm

    def train(self, X, y, epochs=1000, batch_size=32):
        num_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch
            permutation = np.random.permutation(num_samples)
            X_shuffled = X[permutation, :]
            y_shuffled = y[permutation, :]

            with tqdm(total=num_samples, desc=f'Epoch {epoch+1}/{epochs}', unit='sample') as pbar:
                for i in range(0, num_samples, batch_size):
                    # Create mini-batch
                    batch_end = min(i + batch_size, num_samples)
                    X_batch = X_shuffled[i:batch_end, :]
                    y_batch = y_shuffled[i:batch_end, :]

                    # Forward pass
                    output = self.model.forward(X_batch)
                    output = np.clip(output, 1e-9, 1 - 1e-9)

                    # Initial gradient (negative log-likelihood)
                    grad_output = output - y_batch

                    # Backward pass
                    self.model.backward(grad_output, self.learning_rate, self.clip_norm)

                    # Update progress bar
                    pbar.update(batch_end - i)

                    # Calculate and display dynamic loss and accuracy
                    final_output = self.model.forward(X)
                    final_output = np.clip(final_output, 1e-9, 1 - 1e-9)
                    loss = np.mean(-np.log(np.sum(y * final_output, axis=1)))
                    acc = np.mean(np.argmax(y, axis=1) == np.argmax(final_output, axis=1))
                    pbar.set_postfix(loss=loss, accuracy=acc*100)

            # Epoch summary
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {np.round(acc * 100.0, 2)}%")

