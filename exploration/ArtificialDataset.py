import numpy as np
import sklearn.datasets


class ArtificialDataset:
    def __init__(self, n_samples: int, n_feats: int, n_informative: int, seed: int):
        assert 0 < n_informative <= n_feats
        self.__n_informative = n_informative
        self.__informative_feats = n_informative

        self.__rnd = np.random.RandomState(seed=seed)

        self.__X, self.__Y = sklearn.datasets.make_classification(n_samples=n_samples, n_features=n_feats,
                                                                  n_informative=n_informative,
                                                                  n_redundant=0,
                                                                  n_repeated=0,
                                                                  n_classes=2,
                                                                  shuffle=True,
                                                                  random_state=self.__rnd)  # TODO explore other parameters

        self.shuffle()

    @property
    def X(self) -> np.ndarray:
        return self.__X

    @property
    def Y(self) -> np.ndarray:
        return self.__Y

    @property
    def informative_feats(self):
        return self.__informative_feats

    def shuffle(self):
        # shuffle features
        # perm_feats = self.__rnd.permutation(range(self.n_feats))
        # self.__informative_feats = perm_feats==self.__informative_feats
        # self.__X = self.__X[:, perm_feats]

        # shuffle examples
        perm_samples = self.__rnd.permutation(range(self.X.shape[0]))
        self.__X = self.__X[perm_samples, :]
        self.__Y = self.__Y[perm_samples]
