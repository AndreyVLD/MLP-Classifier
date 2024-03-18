from typing import Tuple, List

import numpy as np

from models.network import Network
from models.loss import Loss
from utils import DataUtils


def train_model(network: Network, X_train: np.ndarray, X_valid: np.ndarray, Y_train: np.ndarray, Y_valid: np.ndarray,
                criterion: Loss, num_epochs: int, learning_rate: float, patience: int = 5, shuffle: bool = False) -> \
        tuple[list[float], list[float]]:
    """Trains a neural network model using early stopping for better generalization.

    This function iteratively trains the network on the training set (`X_train`, `Y_train`)
    and evaluates its performance on the validation set (`X_valid`, `Y_valid`).

    Training stops if the validation loss does NOT improve for a specified number of epochs ('patience').

    Args:
        network: The neural network object to train.
        X_train (array-like):  Training input data.
        X_valid (array-like): Validation input data.
        Y_train (array-like): Training target labels.
        Y_valid (array-like):  Validation target labels.
        num_epochs (int): The maximum number of training epochs.
        criterion (Loss): The Loss class used by the model.
        learning_rate (float):  The learning rate for the optimizer.
        patience (int, optional): The number of epochs to wait without improvement
                       in validation loss before stopping training. Defaults to 5.
        shuffle (bool, optional): If True, shuffles training data before each epoch.
                                  Defaults to False.


    Returns:
        tuple: Two lists containing the training and validation losses over epochs.
    """

    losses_train, losses_valid = [], []

    best_valid_loss = float('inf')
    waiting = 0

    # Main Epoch loop
    for epoch in range(num_epochs):
        epoch_loss_train = 0.0
        epoch_loss_valid = 0.0

        print('Epoch', epoch+1)
        if shuffle:
            X_train, Y_train = DataUtils.shuffle_data(X_train, Y_train)

        # Perform Stochastic Gradient Descent
        for i in range(len(X_train)):
            x = X_train[i]
            y_true = Y_train[i]

            # Compute the feedforward
            y_pred = network.forward(x)

            # Compute the loss according to the loss function
            loss, loss_grad = criterion.calculate(y_true, y_pred)
            epoch_loss_train += loss

            # Backpropagation of the loss
            network.backward(loss_grad)

            # Take an optimizer step: Update all weights
            network.optimizer_step(learning_rate)

        # Bookkeeping for the loss during training
        avg_loss_train = epoch_loss_train / len(X_train)
        losses_train.append(avg_loss_train)

        # Validation
        if shuffle:
            X_valid, Y_valid = DataUtils.shuffle_data(X_valid, Y_valid)
        for i in range(len(X_valid)):
            x_valid = X_valid[i]
            y_true_valid = Y_valid[i]

            # Compute the forward pass using the validation set
            y_pred_valid = network.forward(x_valid)

            # Compute the loss for the validation observation
            loss_valid, loss_valid_grad = criterion.calculate(y_true_valid, y_pred_valid)
            epoch_loss_valid += loss_valid

        avg_loss_valid = epoch_loss_valid / len(X_valid)
        losses_valid.append(avg_loss_valid)

        # Early stopping bookkeeping
        if avg_loss_valid < best_valid_loss:
            best_valid_loss = avg_loss_valid
            waiting = 0
        else:
            waiting += 1

        # Early stopping if no improvement in validation loss
        if waiting >= patience:
            break

    return losses_train, losses_valid
