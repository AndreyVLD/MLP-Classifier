import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class MLP:

    def __init__(self):
        """
        Initiates the Multi Layer Perceptron class with a PyTorch Sequential Model
        """
        super().__init__()
        self.test_loader = None
        self.model = nn.Sequential(
            nn.Linear(10, 27),
            nn.LeakyReLU(),
            nn.Linear(27, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        :param X: The Input Array
        :return: The output of the model
        """
        return self.model.forward(X)

    def train(self, features: np.ndarray, targets: np.ndarray, lr: float = 0.01, weight_decay: float = 0.0001,
              batch_size: int = 128, patience_init: int = 8, epochs: int = 200):
        """
        Trains the model
        :param features: input features
        :param targets: output targets in one hot encoding
        :param lr: learning rate
        :param weight_decay: weight decay for the optimizer
        :param batch_size: batch size for the training
        :param patience_init: patience for early stopping
        :param epochs: number of epochs
        """
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.1, random_state=42)
        test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float))
        train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float))

        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")
        patience = patience_init

        for epoch in range(epochs):
            loss = None
            val_loss = 0
            self.model.eval()

            for X, y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                for X, y in train_loader:
                    outputs = self.model(X)
                    val_loss += criterion(outputs, y).item()

            val_loss /= len(train_loader)

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = patience_init  # Reset patience counter
            else:
                patience -= 1
                if patience == 0:
                    print("Early stopping triggered!")
                    break  # Exit the training loop

            print(f"Epoch {epoch + 1}, Loss: {val_loss}")

    def eval_accuracy(self) -> float:
        """
        Evaluates the accuracy of the model
        :return: The accuracy of the model on the test data
        """
        with torch.no_grad():
            correct = 0
            total = 0
            for X, y in self.test_loader:
                outputs = self.model(X)
                _, predicted = torch.max(outputs, 1)
                total += predicted.size(dim=0)
                correct += torch.eq(torch.argmax(y, dim=1), predicted).sum().item()
            return (correct / total) * 100
