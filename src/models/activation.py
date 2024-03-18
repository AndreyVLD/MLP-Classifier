import numpy as np
from abc import ABC, abstractmethod


# Abstract class
class Activation(ABC):
    def __init__(self):
        self.cache = None

    @abstractmethod
    def forward(self, x: np.ndarray):
        """ Perform a forward pass of your activation function.
        Store (cache) the output so it can be used in the backward pass.
        Args:
            x: input to the activation function.

        Returns:
            y: output of the activation function.
        """

        pass

    @abstractmethod
    def backward(self, upstream_gradient: np.ndarray):
        """ Perform a backward pass of the activation function.
        Args:
            upstream_gradient: upstream gradient.

        Returns:
            dx: downstream gradient.
        """

        pass


class LeakyReLU(Activation):
    def __init__(self, c: int):
        super().__init__()
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.where(x > 0, x, self.c * x)
        self.cache = y
        return y

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        dx = np.where(self.cache > 0, 1, self.c) * upstream_gradient
        return dx


class SoftMax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        exps = np.exp(x - np.max(x))
        y = exps / np.sum(exps)
        self.cache = y
        return y

    def backward(self, upstream_gradient: np.ndarray) -> np.ndarray:
        S = self.cache.reshape(-1, 1)

        local_d = np.diagflat(self.cache) - S @ S.T
        dx = upstream_gradient @ local_d
        return dx
