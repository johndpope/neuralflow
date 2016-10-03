import abc


class LearningRateStrategy:
    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def value(self, direction):
        """return the lr symbol"""



class ClippingLearningRate(LearningRateStrategy):

    def __init__(self, thr:float, lr:float):
        self.__lr = lr
        self.__thr = thr

    def value(self, direction):
        result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))