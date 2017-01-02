import numpy as np


class Metrics:
    def __init__(self, predictions: np.array, labels: np.array):
        self.__predictions = predictions
        self.__labels = labels

    @property
    def MAE(self): # TODO refactor
        return np.mean(abs(self.__predictions - self.__labels))

    @property
    def NRMSE(self):
        return 100 * np.sqrt(np.mean((self.__predictions - self.__labels) ** 2)) / np.max(self.__labels)
