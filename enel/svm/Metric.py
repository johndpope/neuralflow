import abc
import numpy as np

from sklearn.metrics import roc_auc_score


class Metric:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute_score(self, predictions: np.array, targets: np.array) -> float:
        """:returns a score"""


class AUCScore(Metric):
    def compute_score(self, predictions: np.array, targets: np.array) -> float:
        return roc_auc_score(y_score=predictions, y_true=targets)


class MAE(Metric):
    def __init__(self, y_scaler=None):
        self.__y_scaler = y_scaler

    def compute_score(self, predictions: np.array, targets: np.array) -> float:
        if self.__y_scaler is not None:
            _predictions = self.__y_scaler.inverse_transform(predictions.ravel())
            _targets = self.__y_scaler.inverse_transform(targets.ravel())
        else:
            _predictions = predictions
            _targets = targets

        return np.mean(abs(_predictions - _targets))
