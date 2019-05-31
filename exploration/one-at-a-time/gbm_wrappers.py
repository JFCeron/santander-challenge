import numpy as np
from sklearn.metrics import roc_curve, auc

def auc_wrapper(individual_preds, t0, agg="like_kernel", w=None, keep=None):
    if agg == "like_kernel":
        final_pred = np.prod(10*individual_preds, axis=1)
    elif agg == "mean":
        final_pred = np.mean(individual_preds, axis=1)
    elif agg == "weighted":
        final_pred = np.mean(individual_preds*w.reshape((1,-1)), axis=1)
    elif agg == "prod_weights":
        final_pred = np.prod(individual_preds**w.reshape((1,-1)), axis=1)
    elif agg == "keep":
        final_pred = np.prod(10*individual_preds[:,keep], axis=1)
    # return AUC of the final prediction
    fpr, tpr, _ = roc_curve(t0, final_pred)
    return auc(fpr, tpr)


class FeatureWiseGBM():
    def __init__(self, params):
        self.params = params
        self.models = []

    def train (self, X_train, t_train):
        """
        Trains a GBM per feature in X and stores all in 'models'
        """
        self.models = []
        features = range(X_train.shape[1])

        for f in features:
            # train a model with just this feature
            lgb_train = lgb.Dataset(X_train.iloc[:,f:f+1], t_train)
            gbm = lgb.train(params, lgb_train, 45, verbose_eval=1000)
            self.models.append(gbm)

    def predict (self, X, agg="like_kernel", w=None, keep=None):
        assert len(models)==X.shape[1], "X's number of columns must equal \
                                        the number of models in FeatureWiseGBM"

        individual_preds = self.predict_individual(X)
        if agg == "like_kernel":
            final_pred = np.prod(10*individual_preds, axis=1)
        elif agg == "mean":
            final_pred = np.mean(individual_preds, axis=1)
        elif agg == "weighted":
            final_pred = np.mean(individual_preds*w.reshape((1,-1)), axis=1)
        elif agg == "prod_weights":
            final_pred = np.prod(individual_preds**w.reshape((1,-1)), axis=1)
        elif agg == "keep":
            final_pred = np.prod(10*individual_preds[:,keep], axis=1)

        return final_pred

    def predict_individual (self, X):
        assert len(models)==X.shape[1], "X's number of columns must equal \
                                        the number of models in FeatureWiseGBM"

        features = range(X.shape[1])
        individual_preds = np.zeros(X_train.shape)
        for f in features:
            gbm = models[f]
            individual_preds[:,f] = gbm.predict(X_train.iloc[:,f:f+1], num_iteration=gbm.best_iteration)
        return individual_preds
