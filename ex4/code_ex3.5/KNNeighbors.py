from builtins import range
from builtins import object
import numpy as np

class KNearestNeighbor(object):
    '''
    Implementation of k-nearest neightnor
    '''

    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        '''
        :param X:
        :param k:
        :return:
        '''
        dists = self._compute_distances(X)
        return self._predict_labels(dists, k=k)

    def _compute_distances(self, X):
        num_test = X.shape[0]
        dists = (np.sum((X**2),axis = 1).reshape(num_test,1) + np.sum(self.X_train**2,axis = 1) - 2*X.dot(self.X_train.T))**0.5
        pass
        return dists

    def _predict_labels(self, dists, k=1):
        '''

        :param dists:
        :param k:
        :return:
        '''
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i,:])[:k]]
            counts = np.bincount(closest_y)
            y_pred[i] = np.argmax(counts)
            pass

        return y_pred
