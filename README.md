# MLP-Classifier
This repository explores a Multi-Layer Perceptron (MLP) classifier, comparing a handcrafted NumPy implementation with a PyTorch version. The goal is to classify shop items into categories based on measurements within a provided dataset.
## Implemented Features
- Handcrafted MLP classifier using only `NumPy`.
- PyTorch MLP classifier.
- Training and evaluation of the classifiers.
- Visualization of the training process and evaluation results.

## Datasets
Datasets are in the `data` folder.
- Measurements of different shop items are stored in the `features.txt` CSV file. The data is stored in a CSV file as a 10 dimensional array.
- The targets are stored in the `targets.txt`. The targets are the categories of the items. The categories are stored as integers from 1 to 7.
- The `unkown.txt` file contains the measurements of the items that need to be classified.
## Code Structure

* `src` - Contains the source code.
* `__init__.py` - The main file to run the code.
* `network.py` - Contains the handcrafted MLP model.
  * `models` - Contains the necessary classes for the handcrafted MLP model.
* `torch_model.py` - Contains the PyTorch MLP model.
* `utils.py` - Contains utility functions for data loading.
* `visualize.py` - Contains functions for visualizing the results.
* `train.py` - Contains functions for training the models.
* `data` - Contains the datasets.

## Usage
- Run the following command to train the MLP model and visualize the results:
- It will first run my handcrafted MLP model and print graphs for losses and confusion matrix and then the PyTorch MLP model.
```bash'
python src\__init__.py
```
### Dependencies

* Python 3.x
* NumPy
* Scikit-learn
* PyTorch
* Matplotlib 

