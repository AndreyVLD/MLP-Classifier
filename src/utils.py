import numpy as np


class DataUtils:

    @staticmethod
    def shuffle_data(inputs: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
        """Shuffles the inputs and labels arrays in unison, maintaining correspondence.

        Args:
            inputs: Input data array.
            labels: Corresponding labels array.
        """
        assert len(inputs) == len(labels)

        permutation = np.random.permutation(len(inputs))
        return inputs[permutation], labels[permutation]

    @staticmethod
    def to_one_hot(labels: list) -> np.ndarray:
        """Converts a list or array of categorical labels into one-hot encoded format.

        Args:
            labels (array-like): An array or list containing the categorical labels
                (often integers representing class indices).

        Returns:
            numpy.ndarray: A matrix where each row represents a label in its one-hot
                encoded form.
        """
        num_classes = len(set(labels))
        one_hot_labels = np.zeros((len(labels), num_classes))
        one_hot_labels[np.arange(len(labels), dtype=int), labels.astype(int) - 1] = 1
        return one_hot_labels
