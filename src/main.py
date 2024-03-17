import numpy as np

from utils import DataUtils


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    features = np.genfromtxt("data/features.txt", delimiter=",")
    targets = np.genfromtxt("data/targets.txt", delimiter=",")

    print(targets)


if __name__ == '__main__':
    main()
