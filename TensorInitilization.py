import abc

import numpy as np


class TensorInitialization(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get(self, size, dtype):
        """initialize a tensor of size='size' with dtype='dtype' """


class GaussianInitialization(TensorInitialization):
    def __init__(self, mean=0, std_dev=0.1, seed: int = 13):
        # random generator
        self.__rnd = np.random.RandomState(seed)
        self.__mean = mean
        self.__std_dev = std_dev

    def get(self, size, dtype="float64"):
        w = np.asarray(self.__rnd.normal(size=size, scale=self.__std_dev, loc=self.__mean), dtype=dtype)
        return w
