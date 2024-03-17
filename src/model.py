import numpy as np


class Linear:
    def __init__(self, in_features, out_features, init_approach):
        """ Randomly initialize the weights and biases.

        Args:
            in_features: number of input features.
            out_features: number of output features.
        """

        self.weight = init_approach((out_features, in_features))
        self.bias = np.zeros(out_features)
        self.init = init_approach
        self.shape = (out_features, in_features)

        self.cache = None
        self.weight_grad = None
        self.bias_grad = None

    def forward(self, x):
        """ Perform the forward pass of a linear layer.
        Store (cache) the input so it can be used in the backward pass.

        Args:
            x: input of a linear layer.

        Returns:
            y: output of a linear layer.
        """
        self.cache = x
        y = self.weight @ x + self.bias

        return y

    def backward(self, upstream_gradient):
        """ Perform the backward pass of a linear layer.

        Args:
            upstream_gradient: upstream gradient.

        Returns:
            dx: downstream gradient.
        """

        dx = self.weight.T @ upstream_gradient

        du = upstream_gradient[np.newaxis].T
        sc = self.cache[np.newaxis]
        self.weight_grad = du @ sc
        self.bias_grad = upstream_gradient

        return dx