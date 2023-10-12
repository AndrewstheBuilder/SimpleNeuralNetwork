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

# Define the layer sizes
layer_sizes = [3, 4, 2]

# NNV = NNVisualizer(2)
# NNV.draw_toy_heatmap()

# Train the Neural Network
# n_h = 9
# parameters = load_parameters()
# parameters = NN.model(X_train, Y_train, parameters, n_h, num_iterations=10000, print_cost=True)

# Initialize new parameters
n_h = 7
parameters = None
lambd = None
parameters = NN.model(X_train, Y_train, parameters, n_h, lambd, num_iterations=10000, print_cost=True)
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
# lambd = 0.4

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

# # Plot histogram for the trained parameters of different regularized models
# plot_histogram_for_reg()
