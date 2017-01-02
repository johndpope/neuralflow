import numpy as np


class Dataset:
    def __init__(self, X: np.array, Y: np.array):
        self.__X = X
        self.__Y = Y

    @property
    def X(self):
        return self.__X

    @property
    def Y(self):
        return self.__Y

    def combine(self, dataset):
        return Dataset(np.concatenate((self.__X, dataset.X)),
                       None if self.__Y is None else np.concatenate((self.__Y, dataset.Y)))
