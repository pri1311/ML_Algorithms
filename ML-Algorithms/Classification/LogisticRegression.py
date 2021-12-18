import numpy as np
from numpy.core.fromnumeric import transpose

class LogisticRegression():
    """
        Class for implementing Multiple Linear Regression using Gradient Descent.
    """

    def __init__(self) -> None:
        self.w = None
        self.b = None

    def sigmoid(self, z):
        """
            Used to calculate regression coefficients
            : param z: scalar or array, 
            : return: scalar or array, sigmoid value of z if it is a scalar, or array of sigmoid value of every element of z if it is an array
        """
        
        return 1.0/(1 + np.exp(-z))

    def cost(self, X, y):
        """
            Used to calculate regression coefficients
            : param X: 2D array, matrix of features, with each row being a data entry
            : param y: array, label or true values
            : return: scalar, cost for particular values of weights and bias
        """

        [m, n] = X.shape

        y = np.array(y).reshape((m, 1))

        y_hat = self.sigmoid(np.dot(X, self.w) + self.b)

        loss = -np.mean(y*(np.log(y_hat)) + (1-y)*np.log(1-y_hat))
        return loss


    def gradients(self, X, y):
        """
            Used to calculate regression coefficients
            : param X: 2D array, matrix of features, with each row being a data entry
            : param y: array, label or true values
            : return: dw and db, gradients for weights and bias respectively
        """
        
        [m, n] = X.shape

        X = np.array(X)
        y = np.array(y).reshape((m, 1))

        hx = self.sigmoid(np.dot(X, self.w) + self.b) 

        db = (1/m) * np.sum(hx - y)
        dw = (1/m) * np.dot(X.T, hx - y)

        return dw, db

    def fit(self, X, y, epochs = 1000, lr = 0.1):
        """
            Used to calculate regression coefficients
            : param X: 2D array, matrix of features, with each row being a data entry
            : param y: array, label or true values
            : param epochs: number, number of iterations
            : param lr: number, learning rate for the model 
            : return: array, cost after each iteration
        """

        [m, n] = X.shape

        self.w = np.zeros((n,1))
        self.b = 0

        cost = []

        for i in range(0, epochs):
            dw, db = self.gradients(X, y)
            self.w -= lr * dw
            self.b -= lr * db

            cost.append(self.cost(X, y))
        return cost

    def predict(self, X):
        """
            Used to predict the value of y
            : param X: 2D array, matrix of features, with each row being a data entry
            : return: array, predicted values
        """

        [m, n] = X.shape
        hx = self.sigmoid((X @ self.w) + self.b)
        pred = np.zeros((m, 1))
        pred[np.where(hx >= 0.5)] = 1
        return pred 

    def accuracy(self, y, y_hat):  
        """
            Used to predict the value of y
            : param y: array, label or true values
            : param y_hat: array, predicted values
            : return: scalar, accuracy of the model
        """

        p = 0
        for i in range(0,len(y)):
            if y[i] == y_hat[i]:
                p += 1
        acc = p / len(y)
        return acc     
    