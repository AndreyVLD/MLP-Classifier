import numpy as np
from src.models.linear import Linear


class Network:
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """ Perform a forward pass over the entire network.

        Args:
            x: input data.

        Returns:
            y: predictions.
        """
        # We forward the output of one layer as input of the next layer, iteratively
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        """ Perform a backward pass over the entire network.

        Args:
            upstream_gradient: upstream gradient.

        Returns:
            dx: downstream gradient.
        """
        # We walk backwards from the last layer
        for layer in reversed(self.layers):
            upstream_gradient = layer.backward(upstream_gradient)
        return upstream_gradient

    def optimizer_step(self, lr: float):
        """ Update the weight and bias parameters of each layer.

        Args:
            lr: learning rate.
        """

        # We iterate through all layers and update the weights and biases
        for layer in self.layers:
            if isinstance(layer, Linear):
                # Update the weights
                layer.weight -= lr * layer.weight_grad
                layer.bias -= lr * layer.bias_grad

                # Reset the gradients
                layer.weight_grad = None
                layer.bias_grad = None

    def reset(self):
        """Resets the weights of Linear layers within
        a model to their initial values."""

        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.weight = layer.init(layer.shape)
