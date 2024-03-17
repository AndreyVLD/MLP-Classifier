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

    @staticmethod
    def split_train_test(features: np.ndarray, targets: np.ndarray, seed: int = None, test_size: float = 0.2) -> (
            np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """Splits feature and target data into training and testing sets.

        This function shuffles the data before splitting to ensure a more representative
        distribution of samples in both the training and testing sets.

        Args:
            features (array-like): The feature data.
            targets (array-like): The corresponding target values.
            seed (int, optional): A random seed for reproducibility. Defaults to None.
            test_size (float, optional): The proportion of data to be included in the
                test set (between 0.0 and 1.0). Defaults to 0.2.

        Returns:
            tuple: A tuple containing:
                * X_train (array-like): The features for the training set.
                * X_test (array-like): The features for the testing set.
                * Y_train (array-like): The targets for the training set.
                * Y_test (array-like): The targets for the testing set.
        """

        if seed is not None:
            np.random.seed(seed)

        # Calculate the number of samples for the test set
        num_test_samples = int(test_size * features.shape[0])

        # Shuffle the indices of the data
        shuffled_indices = np.random.permutation(features.shape[0])

        test_indices = shuffled_indices[:num_test_samples]
        train_indices = shuffled_indices[num_test_samples:]

        X_train, X_test = features[train_indices], features[test_indices]
        Y_train, Y_test = targets[train_indices], targets[test_indices]

        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def split_train_test_val(features: np.ndarray, targets: np.ndarray, seed: int = None, test_size: float = 0.15,
                             val_size: float = 0.15) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                         np.ndarray):
        
        """Splits feature and target data into training, validation, and testing sets.

        This function shuffles the data before splitting to promote a
        more representative distribution of samples across the three sets.

        Args:
            features (array-like): The feature data.
            targets (array-like): The corresponding target values.
            seed (int, optional): A random seed to ensure reproducibility.
                                 Defaults to None.
            test_size (float, optional): The proportion of data for the test set
                (between 0.0 and 1.0). Defaults to 0.15.
            val_size (float, optional): The proportion of data for the validation set
                (between 0.0 and 1.0). Defaults to 0.15.

        Returns:
            tuple: A tuple containing:
                * X_train (array-like): The features for the training set.
                * X_valid (array-like): The features for the validation set.
                * X_test (array-like): The features for the testing set.
                * y_train (array-like): The targets for the training set.
                * y_valid (array-like): The targets for the validation set.
                * y_test (array-like): The targets for the testing set.
        """
        assert test_size + val_size < 1.0

        if seed is not None:
            np.random.seed(seed)

        num_test_samples = int(test_size * features.shape[0])
        num_val_samples = int(val_size * features.shape[0])

        shuffled_indices = np.random.permutation(features.shape[0])

        test_indices = shuffled_indices[:num_test_samples]
        val_indices = shuffled_indices[num_test_samples:num_test_samples + num_val_samples]

        train_indices = shuffled_indices[num_test_samples + num_val_samples:]

        X_train, X_valid, X_test = features[train_indices], features[val_indices], features[test_indices]

        y_train, y_valid, y_test = targets[train_indices], targets[val_indices], targets[test_indices]

        return X_train, X_valid, X_test, y_train, y_valid, y_test
