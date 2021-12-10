import numpy as np
import numpy.linalg as la
import math

verySmallNumber = 1e-14 # That's 1×10⁻¹⁴ = 0.00000000000001


class MultipleLinearRegression():
    """
        Class for implementing Multiple Linear Regression.
    """

    def __init__(self) -> None:
        self.regression_coefficients = None

    def fit(self, X, y):
        """
            Used to calculate regression coefficients
            : param X: 2D array, matrix of features, with each row being a data entry
            : param y: array, label or true values
            : return: None
        """

        X = np.array(X)
        y = np.array(y)

        np.insert(X, 0, 1, 0)
        X = X.astype('float64')
        self.regression_coefficients = la.inv(np.transpose(X) @ X) @ np.transpose(X) @ y

    def predict(self, X):
        """
            Used to predict the value of y
            : param X: 2D array, matrix of features, with each row being a data entry
            : return: array, predicted values
        """

        np.insert(X, 0, 1, 0)
        return X @ self.regression_coefficients

    
class MultipleLinearRegressionGradientDescent(MultipleLinearRegression):
    """
        Class for implementing Multiple Linear Regression with Gradient Descent Method
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y, epochs = 10**3, learning_rate = 10**-5):
        """
            Used to calculate b0(intercept) & b1(slope)
            : param X: 2D array, matrix of features, with each row being a data entry
            : param y: array, label or true values
            : param epochs: number, number of iterations
            : param learning_rate: number, learning rate for the model
            : return: None
        """

        X = np.array(X)
        y = np.array(y)
        n = len(y)
        np.insert(X, 0, 1, 0)
        X = X.astype('float64')
        p = X.shape[1]
        self.regression_coefficients = np.zeros(p)

        for i in range(0, epochs):         
            partial = np.subtract(self.predict(X), y)
            self.regression_coefficients[0] = (np.sum(partial) * 2) / n

            for i in range(1,p):
                partiali = (np.dot(partial , X[:, i]) * 2) / n
                if (math.isnan(partiali)):
                    return
                self.regression_coefficients[i] -= learning_rate *  partiali
        
    


class MultipleLinearRegressionGramSchmidt(MultipleLinearRegression):
    """
        Class for implementing Multiple Linear Regression using Successive Orthogonolization(Gram Schmidt Process/Orthogonolization)
    """

    def __init__(self) -> None:
        super().__init__()
        self.Q = None 
        self.R = None

    def gsBasis(self, X) :
        """
            Used to Orthonormalize matrix X
            : param X: 2D array, matrix of features, with each row being a data entry
            : return: 2D array, column matrix of residuals/ orthonormalized matrix
        """
        
        B = np.array(X, dtype=np.float_) # Make B as a copy of X, since we're going to alter it's values.
        # Loop over all vectors, starting with zero, label them with i
        for i in range(B.shape[1]) :
            # Inside that loop, loop over all previous vectors, j, to subtract.
            for j in range(i) :
                # Complete the code to subtract the overlap with previous vectors.
                # you'll need the current vector B[:, i] and a previous vector B[:, j]
                B[:, i] = B[:, i] - B[:, i] @ B[:, j] * B[:, j]
            # Next insert code to do the normalisation test for B[:, i]
            if la.norm(B[:, i]) > verySmallNumber :
                B[:, i] = B[:, i] / la.norm(B[:, i])
            else :
                B[:, i] = np.zeros_like(B[:, i])
                
            
                
        # Finally, we return the result:
        return B


    def fit(self, X, y):
        """
            Used to calculate regression coefficients, Q and R matrices of QR decomposition
            : param X: 2D array, matrix of features, with each row being a data entry
            : param y: array, label or true values
            : return: None
        """

        np.insert(X, 0, 1, 0)
        X = X.astype('float64')

        self.Q = self.gsBasis(X)
        self.R = np.transpose(self.Q) @ X
        self.regression_coefficients = la.inv(self.R) @ np.transpose(self.Q) @ y
