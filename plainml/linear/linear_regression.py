import numpy as np

class LinearRegression:
    """
    Linear Regression with gradient descent

    Parameters
    ----------
    n_iter : int
        Number of iterations
    learning_rate : float
        Learning rate
    reg : float
        Regularization parameter
    report : bool
        Report the loss after each iteration
    """
    def __init__(self, n_iter=1000, learning_rate=0.00001, reg=0.0, report=True):
        self.weight = None
        self.lr = learning_rate
        self.n_iter = n_iter
        self.report = report
        self.reg = reg
    
    def fit(self, X, y):
        """
        Fit the model according to the given training data
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X.
        """
        self.weight = np.zeros(X.shape[1])
        for _ in range(self.n_iter):
            self.weight -= self.lr * self.gradient_descent(X, y)
            if self.report:
                print('Loss:', self.loss(X, y))
        
    def loss(self, X, y):
        """
        Calculate the loss of the model
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X.
        """
        y_pred = self.predict(X)
        return np.mean(np.square(y_pred - y))
    
    def predict(self, X):
        """
        Predict the target value of X
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        """
        return np.dot(X, self.weight)
    
    def gradient_descent(self, X, y):
        """
        Calculate the gradient descent
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X.
        """
        y_pred = self.predict(X)
        gradient = np.dot(X.T, y_pred - y)
        gradient += self.reg * self.weight
        return gradient

    def accuracy(self, y, y_pred):
        """
        Calculate the accuracy of the model
        
        Parameters
        ----------
        y : array-like, shape = [n_samples]
            Target vector relative to X.
        y_pred : array-like, shape = [n_samples]
            Predicted target vector relative to X.
        """
        correct = 0
        for i in range(len(y)):
            if y.item(i) == y_pred.item(i):
                correct += 1
        return correct / float(len(y)) * 100.0