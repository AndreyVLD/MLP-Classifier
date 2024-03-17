import matplotlib.pyplot as plt


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

        # Plot loss
        ax1.plot(range(1, num_iterations + 1), losses)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Value')
        ax1.set_title(sub_title_1)

        # Plot accuracy
        ax2.plot(range(1, num_iterations + 1), accuracies)
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Value')
        ax2.set_title(sub_title_2)

        fig.suptitle(plot_title)

        plt.tight_layout()
        plt.show()
