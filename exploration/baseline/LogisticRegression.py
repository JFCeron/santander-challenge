"""Regression Model Class Definition"""
import numpy as np
from classification_metrics import roc_auc
import pdb

class LogisticRegression:
    def __init__(self, n_predictors, poly_degree=1, basis_function='poly', regularization=0.0,
                 learning_rate=0.01, threshold=0.5, batch_size=2000):
        self.basis_function = basis_function
        self.M = poly_degree
        self.theta = np.random.randn(poly_degree*n_predictors + 1).reshape((1,-1)) # includes bias term, row vector
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.design_matrix = None
        self.batch_size = batch_size

    def compute_design_matrix(self, X):
        """Returns an N x M Matrix which is the design matrix. Takes a matrix X
           of size N x D and transform it with basis function, which by default
           is the polynomial basis function.
        """
        # You should return an N x M matrix containing the data
        # with the new features depending on the seld.basis_function value
        # pdb.set_trace()
        #if self.basis_function == 'poly':
        self.design_matrix = np.hstack((np.ones((X.shape[0],1)), \
            np.tile(X, self.M)**(np.array([[i]*X.shape[1] for i in range(1,self.M+1)]).flatten()).reshape((1,-1))))

    def optimize(self, X, y, method='gd'):
        """ Updates self.theta value with the new parameter vector wither with
            Normal equation ('normal') or with gradient descent ('gd') which is given
            by method variable.
        """
        m = X.shape[0]
        self.compute_design_matrix(X)

        if method == 'newton':
            # Newthon's method to optimize logisitic Regression
            # Optional.
            print("Newton's method not yet implemented!")
            pass
        elif method == 'gd':
            # store loss at each iteration
            losses = []
            best_loss = np.iinfo("int").max
            consecutive_nondecrease = 0 # count consecutive number of non-decrease steps to stop
            tolerance = (m // self.batch_size)*30 # how many allowed non-decreases
            while(consecutive_nondecrease <= tolerance):
                batch_indices = np.random.choice(m, self.batch_size, False)
                batch_PHI = self.design_matrix[batch_indices,]
                batch_y = y[batch_indices,]
                # cost at this iteration
                y_hat = self.predict(batch_PHI)
                losses.append(self.BCE(y_hat, batch_y))
                # another step of gradient descent
                gradient = np.mean((y_hat - batch_y)*batch_PHI, axis=0).reshape((1,-1))
                gradient[:,1:] = gradient[:,1:] + (self.regularization/m)*self.theta[:,1:] # because the bias is not regularized
                self.theta = self.theta - self.learning_rate*gradient
                # stop if loss has stopped decreasing
                if (best_loss - losses[-1])/best_loss < 0.01:
                    consecutive_nondecrease += 1
                else:
                    best_loss = losses[-1]
                    consecutive_nondecrease = 0
                # dynamically decrease learning rate when training gets stuck
                if consecutive_nondecrease % 5 == 0:
                    self.learning_rate = self.learning_rate*0.9

            return losses

    def predict(self, PHI):
        """ Returns an N x 1 Array containing the predicted values for
            and input array of size N x D (# of data points, Dimensions of input data).
        """
        # Implement a function to make predictions given a set of parameters
        # Remeber the definition of the regression model
        try:
            preds = self.sigmoid(np.sum(PHI*self.theta, axis=1))
        except Exception as e:
            print(e)

        # gradients will have numerical issues if we allow the predictions to be exactly 0 or 1
        preds[preds == 0] = np.finfo(float).resolution
        preds[preds == 1] = 1-np.finfo(float).resolution
        return preds.reshape((-1,1))

    def sigmoid(self, A): # A ndarray
        return 1/(1+np.exp(-A))

    def BCE(self, y_hat, y):
        # we won't regularize the bias term
        return np.mean(-y*np.log(y_hat) - (1-y)*np.log(1-y_hat)) + (self.regularization/(2*y_hat.shape[0]))*np.linalg.norm(self.theta[:,1:])**(self.M)

    def performance(self, X, y):
        """Returns the performance measure for a logistic regression model"""
        # Implement whatever performance measure you think that works
        # For the case try accuracy
        self.compute_design_matrix(X)
        y_hat = self.predict(self.design_matrix) >= self.threshold
        return ( roc_auc(y, y_hat) )
