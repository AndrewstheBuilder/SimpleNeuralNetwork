import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, load_extra_datasets
from simpleneuralnetwork import Model
import json
from init_utils import load_dataset

# Set the random seed
np.random.seed(2)


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
            np.random.randn(N)*1  # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.5  # radius
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
    fig, ax = plt.subplots(figsize=(6, 5))  # width: 5 inches, height: 3 inches
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


def trainOnLR(X, Y):
    # Train the logistic regression classifier
    # Logistic Regression sets a baseline for our Neural Network to beat
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y[0])

    # Plot the decision boundary for logistic regression
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.show()
    plt.title("Logistic Regression")

    # Print accuracy
    LR_predictions = clf.predict(X.T)
    true1s = np.dot(Y, LR_predictions)  # Correctly labeled 1s
    true0s = np.dot(1-Y, 1-LR_predictions)  # Correctly labeled 0s
    print('Accuracy of logistic regression: %d ' % float((true1s + true0s)/float(Y.size)*100) +
          '% ' + "(percentage of correctly labelled datapoints)")


def save_parameters(parameters, file_name='parameters.txt'):
    with open(file_name, 'w') as file:
        # Convert parameters to a JSON-serializable format
        serializable_parameters = {k: v.tolist()
                                   for k, v in parameters.items()}
        file.write(json.dumps(serializable_parameters, indent=4))


def load_parameters(file_name='parameters.txt'):
    with open(file_name, 'r') as file:
        serializable_parameters = json.loads(file.read())
        # Convert parameters back to numpy arrays
        parameters = {k: np.array(v)
                      for k, v in serializable_parameters.items()}
        return parameters


# Load the Data Set
# X, Y = load_planar_dataset()
# noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
# X, Y = noisy_circles
# X, Y = X.T, Y.reshape(1, Y.shape[0])
X_train, Y_train, X_test, Y_test = load_dataset()
print('X_train.shape', X_train.shape)
print('Y_train.shape', Y_train.shape)

# Initialize a Simple Neural Network class with a single hidden layer
NN = Model(X_train, Y_train)

# First Time running then comment out init
# n_x = NN.layer_sizes(X_train, Y_train)[0]
# n_y = NN.layer_sizes(X_train, Y_train)[2]
# parameters = NN.initialize_parameters(n_x, n_y, n_h=9)

# Load parameters if you are NOT running init()
# parameters = load_parameters()

# Train the Neural Network
# parameters = NN.model(X_train, Y_train, parameters, num_iterations=10000, print_cost=True)
# save_parameters(parameters)

# Plot the decision boundary
# plot_decision_boundary(lambda x: NN.predict(parameters, x.T), X_train, Y_train)
# plt.title("Decision Boundary for hidden layer size " + str(4))
# plt.show()

# Print accuracy
# predictions = NN.predict(parameters, X_test)
# Me trying to figure out exactly how that Accuracy Print statement was working
# print('Y_test',Y_test)
# print('Y_test.shape', Y_test.shape)
# print('predictions.T.shape', predictions.T.shape)
# print('(np.dot(Y_test, predictions.T) + np.dot(1 - Y_test, 1 - predictions.T)',
#       (np.dot(Y_test, predictions.T) + np.dot(1 - Y_test, 1 - predictions.T)))
# print('Accuracy: %d' % float((np.dot(Y_test, predictions.T) +
#                              np.dot(1 - Y_test, 1 - predictions.T)) / float(Y_test.size) * 100) + '%')

# Perform Regularization
# Performing L2 Regularization has the following effect on a model:
# [https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/lambda]
# 1. Encourages weight values toward 0 (but not exactly 0)
# 2. Encourages the mean of the weights toward 0, with a normal (bell-shaped or Gaussian) distribution
# Graph these^!
lambd = 0.4

# Load pretrained parameters for Reg Model
# parameters = load_parameters('parameters_L2Reg.txt')

# Train the model with regularization
# parameters = NN.model(X_train, Y_train, parameters, lambd, num_iterations=10000, print_cost=True, L2_reg=True)
# save_parameters(parameters, "parameters_L2Reg_lambd_"+str(lambd)+".txt")

# Plot the decision boundary
# plot_decision_boundary(lambda x: NN.predict(parameters, x.T), X_train, Y_train)
# plt.title("Decision Boundary for hidden layer size " + str(4))
# plt.show()

# predictions = NN.predict(parameters, X_test)

# print('Accuracy: %d' % float((np.dot(Y_test, predictions.T) +
#                               np.dot(1 - Y_test, 1 - predictions.T)) / float(Y_test.size) * 100) + '%')

# Plot Histogram for Different Lambda values
parameters_og = load_parameters('parameters.txt')
parameters_03 = load_parameters('parameters_L2Reg_lambd_0.3.txt')
parameters_04 = load_parameters('parameters_L2Reg_lambd_0.4.txt')
parameters_05 = load_parameters('parameters_L2Reg_lambd_0.5.txt')

datasets = [parameters_og, parameters_03, parameters_04, parameters_05]
x_values = np.arange(1, 10)

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
mean = []

for i, ax in enumerate(axes):
    if i == 4:
        # Print Means
        # You can see that its gradually going towards zero
        print('mean',mean)
        ax.bar(np.arange(1,5), mean, color='skyblue')
        ax.set_title("Means for No L2 Reg to 0.5 lambd Respectively")
    else:
        # Generate Bar charts 1-4
        W2 = datasets[i]["W2"][0]
        W2_abs = np.abs(W2)
        # print('datasets[i]["W2"][0]',datasets[i]["W2"][0])
        # print('W2_abs',W2_abs)
        ax.bar(x_values, W2, color='skyblue')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlim(0, 10)
        if(i == 0):
            ax.set_title(f"No L2 Regularization")
        else:
            ax.set_title(f"Lambd 0.{i+2}")
        ax.set_xlabel("X axis")
        if i == 0:  # Only set the y-label for the first plot for clarity
            ax.set_ylabel("W2 Value")
        mean.append(np.mean(W2_abs))

plt.tight_layout()
plt.show()
