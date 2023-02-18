import numpy as np


class Adaline(object):
    """
    Adaline classifier.

    Parameters:

        eta: floating-point learning rate (in the range 0.0 to 1.0)
        n_iter: number of passes over the training sets
        random_state: seed for the random number generator used to initialize the random weights

    Attributes:

        w_: one-dimensional array of weights after fitting
        cost_: Mean squared error (cost function value) in each epoch.
    """

    def __init__(self, eta: float, n_iter: int, random_state: int):
        self.cost = None
        self.w_ = None
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        :param X: dimensions = [n_samples, n_features] Training vectors where n_samples refers to the number of
        samples and n_features refers to the number of features.;
        :param y: dimensions = [n_samples] Target values
        :return: self
        """
        random_gen = np.random.RandomState(self.random_state)
        self.w_ = random_gen.normal(loc=0.0, scale=0.1, size=1 + X.shape[1])
        self.cost = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
