import numpy as np


class Perceptron(object):
    """
    Perceptron classifier.

    Parameters:

        eta: floating-point learning rate (in the range 0.0 to 1.0)
        n_iter: number of passes over the training sets
        random_state: seed for the random number generator used to initialize the random weights

    Attributes:

        w_: one-dimensional array of weights after fitting
        errors_: number of misclassifications (updates) in each epoch.
    """

    def __init__(self, eta: float, n_iter: int, random_state: int):
        self.w_ = None
        self.errors = None
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
        self.errors = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                if update != 0.0:
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def predict(self, X):
        """
        Computes the net input
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def net_input(self, X):
        """
        Returns the class label after computing the step function
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
