import numpy as np
from src.models.initializers import *

INITIALIZERS = {
    'he_normal_init': he_normal_init,
    'random_init': random_init,
    'uniform_init': uniform_init,
    'lecun_normal_init': lecun_normal_init,
    'xavier_normal_init': xavier_normal_init,
    'xavier_uniform_init': xavier_uniform_init,
    'he_uniform_init': he_uniform_init,
    'orthogonal_init': orthogonal_init
}


class Linear:
    def __init__(self, in_features: int, out_features: int, init_approach='he_normal_init'):
        """ Randomly initialize the weights and biases.

        Args:
            in_features: number of input features.
            out_features: number of output features.
        """
        self.init = INITIALIZERS.get(init_approach, he_normal_init)
        self.weight = self.init((out_features, in_features))
        self.bias = np.zeros(out_features)

        self.shape = (out_features, in_features)

        self.cache = None
        self.weight_grad = None
        self.bias_grad = None

    def forward(self, x: np.ndarray) -> np.ndarray:
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

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        """ Perform the backward pass of a linear layer.

        Args:
            upstream_gradient: upstream gradient.

        Returns:
            dx: downstream gradient.
        """

        dx = self.weight.T @ upstream_gradient

        grad = upstream_gradient[np.newaxis].T
        s_cache = self.cache[np.newaxis]

        self.weight_grad = grad @ s_cache
        self.bias_grad = upstream_gradient

        return dx
