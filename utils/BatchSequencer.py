import numpy as np
from utils.Dataset import BatchProducer


class BatchSequencer(BatchProducer):
    def __init__(self, X, Y, seed: int):
        self.__X = X
        self.__Y = Y
        self.__rnd = np.random.RandomState(seed)

    def get_batch(self, batch_size):
        indexes = self.__rnd.randint(0, self.__X.shape[0], size=(batch_size,))

        batch = {
            'output': self.__Y[indexes],
            'input': self.__X[indexes]
        }

        return batch
