import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron
import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)

X = df.iloc[0:100, [0, 2]].values


# Visualizing data from 2 species of iris and 2 features of them - sepal length and petal length
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')

plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='Versicolor')

plt.xlabel("Sepal length [cm]")
plt.ylabel("Petal length [cm]")
plt.legend(loc='upper left')
plt.show()


ppn = Perceptron(eta=0.1, n_iter=10, random_state=1)

ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Number of actualizations')
plt.show()

# Visualizing decision border

from utils.plotting import plot_decision_regions

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()