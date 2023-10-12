import numpy as np
import copy
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

# Import Utils
from utils.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from utils.visualize_nn import NNVisualizer


class Model:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.NNVisualizer = NNVisualizer(ncols=4)

    def initialize_parameters(self, n_x, n_y, n_h=5):
        """
        Argument:
        n_x -- size of the input layer
        n_y -- size of the output layer
        n_h -- size of the hidden layer

        Returns:
        params -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

    def forward_propagation(self, X, parameters):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialization function)

        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)
        assert(A2.shape == (1, X.shape[1]))

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2,
                 "W1": W1,
                 "W2": W2}

        return A2, cache

    def compute_cost(self, A2, Y):
        """
        Computes the cross-entropy cost given in equation (13)

        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost given equation (13)

        """

        m = Y.shape[1]  # number of examples
        logprobs = np.multiply(np.log(A2), Y) + \
            np.multiply(np.log(1-A2), (1-Y))
        cost = -(1/m) * np.sum(logprobs)

        # makes sure cost is the dimension we expect.
        cost = float(np.squeeze(cost))
        # E.g., turns [[17]] into 17

        return cost

    def compute_cost_with_regularization(self, A2, Y, parameters, lambd):
        """
        Implement the cost function with L2 regularization.
        See formula (2) in Regularization project of Improving Deep Neural Networks Course

        Arguments:
        A2 -- post activation, output of forward propagation, of shape (output size, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        parameters -- python dictionary containing parameters of the model

        Returns:
        cost - value of the regularized loss function (formula(2))
        """
        m = Y.shape[1]
        W1 = parameters["W1"]
        W2 = parameters["W2"]

        # This gives the cross-entropy part of the cost
        cross_entropy_cost = self.compute_cost(A2, Y)

        L2_regularization_cost = (
            np.sum(np.square(W1)) + np.sum(np.square(W2))) * 1/m * lambd/2

        cost = cross_entropy_cost + L2_regularization_cost

        return cost

    def backward_propagation(self, parameters, cache, X, Y):
        """
        Implement the backward propagation using the instructions above.

        Arguments:
        parameters -- python dictionary containing our parameters
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)

        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = X.shape[1]

        # First, retrieve W1 and W2 from the dictionary "parameters".
        W1 = parameters["W1"]
        W2 = parameters["W2"]

        # Retrieve also A1 and A2 from dictionary "cache".
        A1 = cache["A1"]
        A2 = cache["A2"]

        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads

    def backward_propagation_with_regularization(self, X, Y, cache, lambd):
        """
        Implements the backward propagation of our baseline model to which we added an L2 regularization

        Arguments:
        X -- input dataset, of shape (input size, number of examples)
        Y -- "true" labels vector, of shape (output size, number of examples)
        cache -- cache output from forward_propagation()
        lambd -- regularization hyperparameter, scalar

        Returns:
        gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
        """
        m = X.shape[1]  # number of examples
        # First, retrieve W1 and W2 from the dictionary "parameters".
        W1 = cache["W1"]
        W2 = cache["W2"]

        # Retrieve also A1 and A2 from dictionary "cache".
        A1 = cache["A1"]
        A2 = cache["A2"]

        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(dZ2, A1.T) + lambd/m * W2
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = (1/m) * np.dot(dZ1, X.T) + lambd/m * W1
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads

    def update_parameters(self, parameters, grads, learning_rate=1.2):
        """
        Updates parameters using the gradient descent update rule given above

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients

        Returns:
        parameters -- python dictionary containing your updated parameters
        """
        # Retrieve a copy of each parameter from the dictionary "parameters". Use copy.deepcopy(...) for W1 and W2
        W1 = copy.deepcopy(parameters["W1"])
        b1 = parameters["b1"]
        W2 = copy.deepcopy(parameters["W2"])
        b2 = parameters["b2"]

        # Retrieve each gradient from the dictionary "grads"
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        # Update rule for each parameter
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

    def layer_sizes(self, X, Y):
        """
        Arguments:
        X -- input dataset of shape (input size, number of examples)
        Y -- labels of shape (output size, number of examples)

        Returns:
        n_x -- the size of the input layer
        n_h -- the size of the hidden layer
        n_y -- the size of the output layer
        """
        n_x = X.shape[0]
        n_h = 4
        n_y = Y.shape[0]
        return (n_x, n_h, n_y)

    def model(self, X, Y, parameters, n_h, lambd, num_iterations=10000, print_cost=False):
        """
        Arguments:
        X -- dataset of shape (2, number of examples)
        Y -- labels of shape (1, number of examples)
        n_h -- size of the hidden layer
        num_iterations -- Number of iterations in gradient descent loop
        print_cost -- if True, print the cost every 1000 iterations

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(3)
        n_x = self.layer_sizes(X, Y)[0]
        n_y = self.layer_sizes(X, Y)[2]

        if(parameters == None):
            # Initialize parameters
            parameters = self.initialize_parameters(n_x, n_y, n_h)

        self.NNVisualizer.draw_heatmap(Ws=[parameters["W1"], parameters["W2"].T], Bs=[
                                       parameters["b1"], parameters["b2"]])
        # Loop (gradient descent)
        for i in range(0, num_iterations):
            A2, cache = self.forward_propagation(X, parameters)
            if(lambd != None):
                cost = self.compute_cost_with_regularization(
                    A2, Y, parameters, lambd)
                grads = self.backward_propagation_with_regularization(
                    X, Y, cache, lambd)
            else:
                cost = self.compute_cost(A2, Y)
                grads = self.backward_propagation(parameters, cache, X, Y)
            parameters = self.update_parameters(parameters, grads)

            # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if i % 5000 == 0:
                self.NNVisualizer.draw_heatmap(Ws=[parameters["W1"], parameters["W2"].T], Bs=[
                                parameters["b1"], parameters["b2"]])

        return parameters

    def predict(self, parameters, X):
        """
        Using the learned parameters, predicts a class for each example in X

        Arguments:
        parameters -- python dictionary containing your parameters
        X -- input data of size (n_x, m)

        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """

        # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        A2, cache = self.forward_propagation(X, parameters)
        predictions = (A2 > 0.5)

        return predictions
