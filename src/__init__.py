import numpy as np

from train import train_model
from utils import DataUtils
from models.linear import Linear
from models.activation import LeakyReLU, SoftMax
from models.network import Network
from models.loss import CategoricalCrossEntropy
from visualization import ModelVisualizer


def confusion_matrix(net, X_test, Y_test):
    test_pred = DataUtils.get_predictions(net, X_test)
    conf_matrix = DataUtils.confusion_matrix(Y_test, test_pred, num_classes=7)
    print(conf_matrix)
    ModelVisualizer.plot_confusion_matrix(conf_matrix)


def main():
    features = np.genfromtxt("data/features.txt", delimiter=",")
    targets = DataUtils.to_one_hot(np.genfromtxt("data/targets.txt", delimiter=","))

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = DataUtils.split_train_test_val(features, targets, seed=42)

    layers = [Linear(10, 27, 'xavier_uniform_init'), LeakyReLU(0.01),
              Linear(27, 7, 'xavier_uniform_init'), SoftMax()]

    net = Network(layers)
    losses_train, losses_valid = train_model(net, X_train, X_valid, Y_train,
                                             Y_valid, CategoricalCrossEntropy(), 40, 0.01, patience=8)

    ModelVisualizer.plot_results(losses_train, losses_valid, 'Train Loss',
                                 'Validation Loss', 'Overfit Plot')

    print('Accuracy on the test data', DataUtils.evaluate_accuracy(net, X_test, Y_test))
    confusion_matrix(net, X_test, Y_test)


if __name__ == '__main__':
    main()
