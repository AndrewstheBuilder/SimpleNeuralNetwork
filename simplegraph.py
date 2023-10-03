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
    N = int(m/4)  # number of points per class
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

def visualize_dataset(X, Y):
    print('X', X.shape)
    print('Y', Y.shape)

    # Visualize the distribution of correct labels
    sub_data = Y[0]
    count_0 = np.sum(sub_data == 0)
    count_1 = np.sum(sub_data == 1)
    labels = ['0', '1']
    counts = [count_0, count_1]
    print('counts_0', count_0)
    print('count_1', count_1)
    fig, ax = plt.subplots(figsize=(6,5)) # width: 5 inches, height: 3 inches
    bars = ax.bar(labels, counts, color=['red', 'blue'])
    ax.set_title('Count of 0s and 1s in Specified Ranges')
    ax.set_xlabel('Element')
    ax.set_ylabel('Count')
    # Annotating the bars with the count
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate('{}'.format(count),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),    # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Visualize the training/dev data in a scatter plot:
    fig, ax1 = plt.subplots()
    ax1.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)

    plt.show()

X, Y = load_planar_dataset()
visualize_dataset(X, Y)

# Run the perceptron on this dataset
X[0, :]
