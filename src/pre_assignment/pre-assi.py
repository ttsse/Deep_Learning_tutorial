import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_auto():

	# import data
	Auto = pd.read_csv('../../datasets/Auto.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()
	Auto.columns
	# Extract relevant input and output for traning
	X_train = Auto.drop(columns='mpg')
	Y_train = Auto[['mpg']]

	return X_train, Y_train


def normalize(X):

    # apply normalization techniques on the samples
    for column in X.columns:
        X[column] = (X[column] - X[column].mean()) / np.sqrt(np.mean((X[column] - X[column].mean())**2))
            
    return X


class LinearRegressionGD:
    """Linear Regression Using Gradient Descent.
    Parameters
    ----------
    lr : float
        Learning rate
    n_step : int
        No of passes over the training set
    Attributes
    ----------
    w_ : array-like, shape = [n_features, 1]
        weights
    b_: array-like, shape = [1, ]
        biases
    cost_ : list
        total error of the model after each iteration
    """

    def __init__(self, lr=0.1, n_step=1000, n_feature=1):
        """Initialize the weights and biases
        """
        self.lr = lr
        self.n_step = n_step
        self.w_ = np.zeros((n_feature, 1))
        self.b_ = np.zeros(1,)
        self.cost_ = []

    def fit(self, x, y):
        """Fit the training data
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        Returns
        -------
        self : object
        """

        n = x.shape[0]
        for _ in range(self.n_step):
            y_pred = np.dot(x, self.w_) + self.b_
            residuals = y_pred - y
            
            # calculate the cost
            cost = np.sum((residuals ** 2)) / n 
            self.cost_.append(cost)

            # calculate the gradients for weights and biases
            gradient_vector_w = np.dot(x.T, residuals)
            gradient_vector_b = np.mean(residuals)

            # update weights and biases with gradients
            self.w_ -= (self.lr / n) * gradient_vector_w
            self.b_ -= self.lr * gradient_vector_b

        return self

    def predict(self, x):
        """ Predicts the value for sample x
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        return np.dot(x, self.w_) + self.b_
    
def main():
    X_train, Y_train= load_auto()
    
    # LR model 1
    X = X_train[['horsepower']]
    Y = Y_train.values

    X = normalize(X)
    num_feture = X.shape[1]
    LR1 = LinearRegressionGD(n_feature=num_feture)
    LR1.fit(X, Y)
    y_pred = LR1.predict(X)

    plt.figure()
    plt.plot(X_train[['horsepower']], Y, 'r.', label="y_true")
    plt.plot(X_train[['horsepower']], y_pred, 'g-', label="y_pred")
    plt.legend()
    plt.xlabel('horsepower')
    plt.ylabel('mpg')
    plt.title('true mpg vs. predicted mpg')
    plt.savefig("prediction.png")

    print(f"The final cost value is {LR1.cost_[-1]}")
    plt.figure()
    plt.plot(LR1.cost_, 'r.')
    plt.xlabel('iteration')
    plt.ylabel('Cost(J)')
    plt.savefig("cost1.png")
    # LR model 2

    X = X_train.drop(columns='name')
    Y = Y_train.values

    X = normalize(X)
    num_feture = X.shape[1]
    LR2 = LinearRegressionGD(n_feature=num_feture)
    LR2.fit(X, Y)

    print(f"The final cost value is {LR2.cost_[-1]}")
    plt.figure()
    plt.plot(LR2.cost_, 'b.')
    plt.xlabel('iteration')
    plt.ylabel('Cost(J)')
    plt.savefig("cost2.png")

if __name__ == '__main__':
     main()