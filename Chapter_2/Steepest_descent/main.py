import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from steepest_descent import AdalineSGD
from utils.plotting import plot_decision_regions

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

Y = df.iloc[0:100, 4].values

Y = np.where(Y == "Iris-setosa", -1, 1)

X = df.iloc[0:100, [0, 2]].values

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, Y)
plot_decision_regions(X_std, Y, classifier=ada)
plt.title("Adaline - Stochastic gradient descent.")
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost) + 1), ada.cost, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Mean cost")
plt.show()
