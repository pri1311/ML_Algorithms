import numpy as np
import math

class SimpleLinearRegression():
    """
        Class for implementing Simple Linear Regression.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        """
            Used to calculate b0(intercept) & b1(slope)
            : param X: array, single feature
            : param y: array, label or true values
            : return: None
        """

        XMean = np.mean(X)
        yMean = np.mean(y)

        self.slope = np.dot((X - XMean), (y - yMean)) / np.sum((X - XMean) ** 2)
        self.intercept = yMean - (self.slope * XMean)

    def predict(self, x):
        """
            Used to predict the value of y
            : param X: array, single feature
            : return: array, predicted values
        """

        return self.intercept + (self.slope * x)


class SimpleLinearRegressionGradientDescent(SimpleLinearRegression):
    """
        Class for implementing Simple Linear Regression with Gradient Descent Method
    """

    def __init__(self, slope = 0, intercept = 0) -> None:
        self.slope = slope
        self.intercept = intercept

    def fit(self, X, y, epochs = 10**6, learning_rate = 10**-5):
        """
            Used to calculate b0(intercept) & b1(slope)
            : param X: array, single feature
            : param y: array, label or true values
            : param epochs: number, number of iterations
            : param learning_rate: number, learning rate for the model
            : return: None
        """

        X = np.array(X)
        y = np.array(y)
        n = len(y)

        for i in range(0, epochs):            
            partial = np.subtract(self.predict(X), y)
            partial0 = (np.sum(partial) * 2) / n
            partial1 = (np.dot(partial , X) * 2) / n

            if math.isnan(partial0) or math.isnan(partial1) :
                break
        
            self.slope -= learning_rate * partial1
            self.intercept -= learning_rate * partial0
    

    