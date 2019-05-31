"""
many classification metrics for truths y, scores y_hat and threshold t
    precision & recall
    area under ROC (AUC)
"""
import numpy as np
from sklearn.metrics import roc_auc_score

def precision(y, y_hat, t):
    tp = tp(y, y_hat, t)
    fp = fp(y, y_hat, t)
    return tp/(tp + fp)

def recall(y, y_hat, t):
    tp = tp(y, y_hat, t)
    fn = fn(y, y_hat, t)
    return tp/(tp + fn)

def roc_auc(y, y_hat):
    return roc_auc_score(y, y_hat)

def tp(y, y_hat, t):
    pred = y_hat > t
    return np.sum(y & pred)

def fp(y, y_hat, t):
    pred = y_hat > t
    return np.sum((y==0) & pred)

def tn(y, y_hat, t):
    pred = y_hat > t
    return np.sum((y==0) & (pred==0))

def fn(y, y_hat, t):
    pred = y_hat > t
    return np.sum(y & (pred==0))

def auc_wrapper(individual_preds, t0, agg="like_kernel", w=None):
    if agg == "like_kernel":
        final_pred = np.prod(10*individual_preds, axis=1)
    elif agg == "mean":
        final_pred = np.mean(individual_preds, axis=1)
    elif agg == "weighted":
        final_pred = np.mean(individual_preds*w.reshape((1,-1)), axis=1)
    elif agg == "prod_weights":
        final_pred = np.prod(individual_preds**w.reshape((1,-1)), axis=1)
        print(final_pred.shape)
    # return AUC of the final prediction
    fpr, tpr, _ = roc_curve(t0, final_pred)
    return auc(fpr, tpr)
