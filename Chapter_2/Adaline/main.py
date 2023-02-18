import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from adaline import Adaline

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

Y = df.iloc[0:100, 4].values

Y = np.where(Y == "Iris-setosa", -1, 1)

X = df.iloc[0:100, [0, 2]].values

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
# adal1 = Adaline(n_iter=10, eta=0.01, random_state=1).fit(X, Y)
# ax[0].plot(range(1, len(adal1.cost) + 1), np.log10(adal1.cost), marker='o')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('Log (Mean Squared Error)')
# ax[0].set_title('Adaline - eta equal to 0.01')
#
# adal2 = Adaline(n_iter=10, eta=0.0001, random_state=1).fit(X, Y)
# ax[1].plot(range(1, len(adal2.cost) + 1), adal2.cost, marker='o')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('Log (Mean Squared Error)')
# ax[1].set_title('Adaline - eta equal to 0.0001')
#
# plt.show()


# Standardized Adaline


X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std [:,1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

adal2 = Adaline(n_iter=10, eta=0.01, random_state=1).fit(X_std, Y)

from utils.plotting import plot_decision_regions

plot_decision_regions(X_std, Y, classifier=adal2)

plt.title("Adaline - simple gradient")
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(adal2.cost) + 1), adal2.cost, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.show()