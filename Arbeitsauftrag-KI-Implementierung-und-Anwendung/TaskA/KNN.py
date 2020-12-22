import numpy as np
from collections import Counter


def euc_dist(a, b):
    return np.linalg.norm(a - b, ord=2)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def norm_data(X):
    mean, std = X.mean(axis=0), X.std(axis=0)
    return (X - mean) / std, (mean, std)


def argsort(a):
    return np.array(a).argsort()


class KNN:
    def __init__(self, k=5, dist_metric='euclidean', norm=True):
        self.k = k
        self.isFit = False  # model fitting done?
        self.norm = norm
        self._set_dist_func(dist_metric)

    def fit(self, X_train, y_train, v=False):
        self.X_train = X_train
        self.y_train = y_train

        y_train_pred, y_train_pred_proba = [], []
        for i, x_i in enumerate(X_train):
            distances = []
            for j, x_j in enumerate(X_train):
                if i == j:
                    dist_ij = 0
                else:
                    dist_ij = self.dist_func(x_i, x_j)

                distances.append(dist_ij)
            pred_i = self.estimate_point(distances, y_train)
            y_train_pred_i, y_train_pred_proba_i = pred_i
            y_train_pred.append(y_train_pred_i)
            y_train_pred_proba.append(y_train_pred_proba_i)

        if v:
            trn_acc = accuracy(y_train, y_train_pred)
            print('training accuracy: {}'.format(trn_acc))
        self.isFit = True

    def estimate_point(self, distances, y):
        sort_idx = argsort(distances)
        y_closest = y[sort_idx][:self.k]
        most_common = Counter(y_closest).most_common(1)[0]
        y_pred_i = most_common[0]
        y_pred_proba_i = most_common[1] / len(y_closest)
        return y_pred_i, y_pred_proba_i

    def predict(self, X_new):

        y_new_pred, y_new_pred_proba = [], []
        for i, x_i in enumerate(X_new):
            distances = []
            for j, x_j in enumerate(self.X_train):
                dist_ij = self.dist_func(x_i, x_j)
                distances.append(dist_ij)

            pred_i = self.estimate_point(distances, self.y_train)
            y_pred_i, y_pred_proba_i = pred_i
            y_new_pred.append(y_pred_i)
            y_new_pred_proba.append(y_pred_proba_i)
        return y_new_pred

    def _set_dist_func(self, dist_metric):

        implemented_metrics = {'euclidean': euc_dist, }
        self.dist_func = implemented_metrics[dist_metric]
