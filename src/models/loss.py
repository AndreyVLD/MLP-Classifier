import numpy as np
from abc import ABC, abstractmethod


# Abstract
class Loss(ABC):
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
        """Calculates the loss value and its gradient."""
        pass


class CategoricalCrossEntropy(Loss):

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
        y_pred = np.clip(y_pred, a_min=1e-7, a_max=1 - 1e-7)
        grad = -(y_true / y_pred)
        loss = - (y_true @ np.log(y_pred))

        return loss, grad


class MeanSquaredError(Loss):

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
        grad = 2 * (y_pred - y_true) / y_true.shape[1]
        loss = np.mean(np.square(y_pred - y_true))

        return loss, grad
