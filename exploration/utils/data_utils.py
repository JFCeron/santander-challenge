"""Data utils module"""
import numpy as np

def split_data(X, t, w=[0.7, 0.15, 0.15]):
    """Takes and array of N x D (N: amount of data points, D: dimension)
       and returns three arrays containing Training, validation and test
       sets
    """
    # normalize train/val/test weights vector
    w = np.array(w)
    w = w/np.sum(w)
    # train/val/test indices
    train_i, val_i, test_i = split_data_indices(X.shape[0], w)
    return X[train_i,], t[train_i], X[val_i,], t[val_i], X[test_i,], t[test_i,]

def split_data_indices(N, w=[0.7, 0.15, 0.15]):
    """Splits the indices [1, .., N] into train, test and validation subsets
    """
    # normalize train/val/test weights vector
    w = np.array(w)
    w = w/np.sum(w)
    # train/val/test indices
    indices = np.random.multinomial(n=1, pvals=w, size=N)==1
    return indices[:,0], indices[:,1], indices[:,2]

def random_batch(X, batch_size):
    batch_indices = np.random.choice(X.shape[0], batch_size, False)
    return X[batch_indices,]
