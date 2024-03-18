import itertools

import matplotlib.pyplot as plt
import numpy as np


class ModelVisualizer:

    @staticmethod
    def plot_results(losses, accuracies, sub_title_1, sub_title_2, plot_title):
        """Plots training loss and accuracy over iterations in 2 subplots.

        Args:
            :param losses: A list of training losses over iterations.
            :param accuracies: A list of training accuracies over iterations.
            :param plot_title: The title of the whole plot.
            :param sub_title_2: The title of the second sub plot.
            :param sub_title_1: The title of the first subplot.
        """

        num_iterations = len(losses)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        ax1.plot(range(1, num_iterations + 1), losses)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Value')
        ax1.set_title(sub_title_1)

        ax2.plot(range(1, num_iterations + 1), accuracies)
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Value')
        ax2.set_title(sub_title_2)

        fig.suptitle(plot_title)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(matrix: np.ndarray, num_classes: int = 7):
        """Visualizes a confusion matrix using a heatmap.

        Args:
            :param matrix: (numpy.ndarray) The confusion matrix to plot.
            :param num_classes: Number of classes in the matrix
        """

        classes = list(str(range(num_classes)))
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()

        tick_marks = np.arange(len(matrix))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, str(matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > matrix.max() / 2. else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        plt.show()
