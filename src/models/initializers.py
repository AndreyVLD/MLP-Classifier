import numpy as np
import math


def random_init(shape: (int, int)) -> np.ndarray:
    """
        Args:
            shape (tuple): the shape of the weights for one layer
        Returns:
            numpy.ndarray: The initialized weight matrix
    """
    return np.random.randn(*shape)


def uniform_init(shape: (int, int)) -> np.ndarray:
    return np.random.uniform(-0.1, 0.1, shape)


def he_normal_init(shape: (int, int)) -> np.ndarray:
    return np.random.randn(*shape) * np.sqrt(2 / shape[1])


def lecun_normal_init(shape: (int, int)) -> np.ndarray:
    return np.random.randn(*shape) * np.sqrt(1 / shape[1])


def xavier_normal_init(shape: (int, int)) -> np.ndarray:
    return np.random.randn(*shape) * np.sqrt(2 / (shape[1] + shape[0]))


def xavier_uniform_init(shape: (int, int)) -> np.ndarray:
    val_range = math.sqrt(6) / math.sqrt(shape[0] + shape[1])
    return np.random.uniform(-val_range, val_range, shape)


def he_uniform_init(shape: (int, int)) -> np.ndarray:
    limit = math.sqrt(6 / shape[1])
    return np.random.uniform(-limit, limit, shape)


def orthogonal_init(shape: (int,int)) -> np.ndarray:
    random_matrix = np.random.randn(*shape)
    U, _, Vt = np.linalg.svd(random_matrix, full_matrices=False)
    if shape[0] > shape[1]:
        return U
    else:
        return Vt
