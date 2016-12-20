import abc
import numpy as np
import scipy
import sklearn
from sklearn.feature_selection import VarianceThreshold


class FeatureSelection:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def select(self, X):
        """"""


class FeatureSelectionStrategy:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, X, Y) -> FeatureSelection:
        """"""


class IndexBasedFeatureSelection(FeatureSelection):
    def __init__(self, indexes: np.array):
        self.__indexes = indexes

    def select(self, X):
        return X[:, self.__indexes]


class SkLearnAdapter(FeatureSelection):
    def __init__(self, sk_learn_obj):
        self.__obj = sk_learn_obj

    def select(self, X):
        return self.__obj.transform(X)


class NullFeatureSelectionStrategy(FeatureSelectionStrategy):
    def train(self, X, Y) -> FeatureSelection:
        return IndexBasedFeatureSelection(np.arange(X.shape[1]))


class PrecomputedFeatureSelectionStrategy(FeatureSelectionStrategy):
    def __init__(self, csv: str):
        self.__csv = csv

    def train(self, X, Y) -> FeatureSelection:
        file = open(self.__csv, "r")
        content = file.read()
        file.close()
        values = content.split(",")
        indexes = np.array([int(s.strip()) for s in values])
        print(indexes.shape)
        return IndexBasedFeatureSelection(indexes)


class VarianceThresholdStrategy(FeatureSelectionStrategy):
    def __init__(self, thr=.8 * (1 - .8)):
        self.__thr = thr

    def train(self, X, Y) -> FeatureSelection:
        # k = int(X.shape[1] * keep_ratio)
        k = 1632

        std = np.std(X, axis=0)
        sorted_indexes = np.argsort(std)

        selected = sorted_indexes[-k-1:-1]

        return IndexBasedFeatureSelection(selected)

        #
        # obj = VarianceThreshold(threshold=(self.__thr))
        # obj.fit(X)
        # return SkLearnAdapter(obj)


# class PerasonFeatureSelectionStrategy(FeatureSelectionStrategy):
#     def __init__(self, n):
#         self.__n = n
#
#     def train(self, X, Y) -> FeatureSelection:
#         c, p = scipy.stats.pearsonr(X, y)
#
#         indexes = np.argsort(c)
#         return FeatureSelection(indexes[:self.__n])


if __name__ == "__main__":
    f = PrecomputedFeatureSelectionStrategy(csv="/media/homegalvan/EnelNew/feature_sel/cbf98865.csv")
    a = f.train(None, None)
