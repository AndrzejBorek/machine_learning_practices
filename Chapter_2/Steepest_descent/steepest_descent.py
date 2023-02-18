import numpy as np
from numpy.random import seed


class AdalineSGD(object):
    """

    Parameters:

        eta: floating-point learning rate (in the range 0.0 to 1.0)
        n_iter: number of passes over the training sets
        random_state: seed for the random number generator used to initialize the random weights
        shuffle: boolean, by default True. If it's true, learning set of data will be shuffled before each epoch to
        avoid model being stuck in local minimum.
    """

    def __init__(self, eta, n_iter, shuffle=True, random_state=None):
        self.rgen = None
        self.w_ = None
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        """
             :param X: dimensions = [n_samples, n_features] Training vectors where n_samples refers to the number of
             samples and n_features refers to the number of features.;
             :param y: dimensions = [n_samples] Target values
             :return: self
        """
        self._initialize_weights(X.shape[1])
        self.cost = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """
        Adjusts the training data without re-initializing the weights
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target) -> float:
        """
        Uses Adaline to weight actualizations
        """
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:] + self.w_[0])

    def activation(self, X):
        """
        Calculate the linear activation function.
        """
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
