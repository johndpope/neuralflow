import abc

import numpy as np
from enel.svm.Dataset import Dataset


class Model:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def predict(self, X: np.array) -> np.array:
        """:returns an array of predictions for the feature matrix X. number of examples is X.shape[0]"""


class TrainingStrategy:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, training_set:Dataset, validation_set: Dataset = None):
        """returns a trained classifier"""
