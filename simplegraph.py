from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np

# Set the seed
np.random.seed(1)
# random.seed(123456)

# Constants
WIDTH, HEIGHT = 600, 600

# Randomly generate line's slope and intercept
m = np.random.uniform(-1, 1)
b = np.random.uniform(HEIGHT // 4, HEIGHT // 2)

# define a Perceptron
ptron = Perceptron(3, HEIGHT)


def load_planar_dataset():
    """
    This is from the planar_utils implementation on [https://www.kaggle.com/code/kolisnehar/planar-utils]
    """
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m/2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    # labels vector (0 for red, 1 for blue)
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N*j, N*(j+1))
        t = np.linspace(j*3.12, (j+1)*3.12, N) + \
            np.random.randn(N)*0.2  # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2  # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


X, Y = load_planar_dataset()
print('X', X.shape)
print('Y', Y.shape)

