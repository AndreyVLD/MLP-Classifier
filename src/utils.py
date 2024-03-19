import numpy as np
from models.network import Network


class DataUtils:

    @staticmethod
    def shuffle_data(inputs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Shuffles the inputs and labels arrays in unison, maintaining correspondence.

        Args:
            inputs: Input data array.
            labels: Corresponding labels array.
        """
        assert len(inputs) == len(labels)

        permutation = np.random.permutation(len(inputs))
        return inputs[permutation], labels[permutation]

    @staticmethod
    def to_one_hot(labels: np.ndarray) -> np.ndarray:
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
    def split_train_test(features: np.ndarray, targets: np.ndarray, seed: int = None, test_size: float = 0.2) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        num_test_samples = int(test_size * features.shape[0])
        shuffled_indices = np.random.permutation(features.shape[0])

        test_indices = shuffled_indices[:num_test_samples]
        train_indices = shuffled_indices[num_test_samples:]

        X_train, X_test = features[train_indices], features[test_indices]
        Y_train, Y_test = targets[train_indices], targets[test_indices]

        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def split_train_test_val(features: np.ndarray, targets: np.ndarray, seed: int = None, test_size: float = 0.15,
                             val_size: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                              np.ndarray, np.ndarray]:

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

    @staticmethod
    def k_fold_cross_validation(features: np.ndarray, targets: np.ndarray, k: int = 2, seed: int = None) -> (
            list)[(np.ndarray, np.ndarray, np.ndarray, np.ndarray)]:
        """Performs k-fold cross-validation on feature and target data.

        This function splits the data into 'k' folds, then iteratively uses one fold as
        the validation set and the remaining folds as the training set. This allows for
        more robust model evaluation across different splits of the dataset.

        Args:
            features (array-like): The feature data.
            targets (array-like): The corresponding target values.
            k (int, optional): The number of folds to create. Defaults to 2.
            seed (int, optional): A random seed for reproducibility. Defaults to None.

        Returns:
            list: A list of tuples. Each tuple contains:
                * X_train_fold (array-like):
                        Features for the training set in the current fold.
                * X_val_fold (array-like):
                        Features for the validation set in the current fold.
                * y_train_fold (array-like):
                        Targets for the training set in the current fold.
                * y_val_fold (array-like):
                        Targets for the validation set in the current fold.
        """
        assert k > 0

        if seed is not None:
            np.random.seed(seed)

        shuffled_indices = np.random.permutation(features.shape[0])

        fold_size = len(shuffled_indices) // k
        folds = [shuffled_indices[i * fold_size: (i + 1) * fold_size] for i in range(k)]

        k_fold_data = []
        if k > 0:
            for i in range(k):
                val_indices = folds[i]
                train_indices = np.concatenate([fold for j, fold in enumerate(folds) if j != i])

            X_train_fold, X_val_fold = features[train_indices], features[val_indices]
            y_train_fold, y_val_fold = targets[train_indices], targets[val_indices]

            k_fold_data.append((X_train_fold, X_val_fold, y_train_fold, y_val_fold))

        return k_fold_data

    @staticmethod
    def evaluate_accuracy(network: Network, test_data: np.ndarray, true_labels: np.ndarray) -> float:

        """Calculates the accuracy of a trained neural network on a test dataset.

        Accuracy is determined by comparing the network's predicted class labels to the
        true class labels of the test data.

        Args:
            network: A trained neural network object.
            test_data (array-like): The input data to evaluate the network on.
            true_labels (array-like): The true target labels corresponding to the test data.

        Returns:
            float: The accuracy of the network on the test set (between 0.0 and 1.0).
        """

        correct = 0
        total = 0
        for x, y in zip(test_data, true_labels):
            predicted = network.forward(x)

            pred_class = np.argmax(predicted) + 1
            true_class = np.argmax(y) + 1
            correct += (pred_class == true_class)
            total += 1

        accuracy = correct / total
        return accuracy

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> np.ndarray:
        """Calculates the confusion matrix for evaluating classification performance.

        Args:
            y_true (array-like): The true class labels.
            y_pred (array-like): The predicted class labels.
            num_classes (int, optional): The total number of classes. Defaults to 7.

        Returns:
            numpy.ndarray: A square matrix where rows represent true labels and
                columns represent predicted labels. Each element counts the number of
                instances where a true label was predicted as a certain class.
        """

        matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

        for true_label, pred_label in zip(y_true, y_pred):
            true_lb = np.argmax(true_label) + 1
            matrix[true_lb - 1][pred_label - 1] += 1

        return matrix

    @staticmethod
    def get_predictions(network, test_data) -> np.ndarray:
        """Generates class predictions for a set of test data using
           a trained neural network.

        Args:
            network: A trained neural network object.
            test_data (array-like): The input data to generate predictions for.

        Returns:
            list: A list of predicted class labels (integers).
        """

        predictions = []
        for x in test_data:
            predicted = network.forward(x)
            pred_class = np.argmax(predicted) + 1
            predictions.append(pred_class)
        return np.array(predictions)
