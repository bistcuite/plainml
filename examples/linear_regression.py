from plainml.linear import *
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

lr = LinearRegression(n_iter=20, learning_rate=0.1,reg=1)
lr.fit(X, y)
print(lr.accuracy(X, y))
