import numpy as np
from abc import ABC, abstractmethod


# Abstract
class Loss(ABC):
    @abstractmethod
    def calculate(self, y_true, y_pred):
        """Calculates the loss value and its gradient."""
        pass
