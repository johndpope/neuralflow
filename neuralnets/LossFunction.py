import abc

import tensorflow as tf


class LossFunction:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def value(self, y, t):
        """returns the (symbolic) value of the objective function given a value 'y' and the targets 't'"""


class CrossEntropy(LossFunction):
    def __init__(self, single_output: bool = False):
        self.__single_output = single_output

    def value(self, y, t):
        if self.__single_output:
            c = -tf.reduce_sum((t * tf.log(y + 1e-10) + (1 - t) * tf.log((1 - y) + 1e-10)), reduction_indices=[1])
        else:
            c = -tf.reduce_sum(t * tf.log(y), reduction_indices=[1])
        return tf.reduce_mean(c)


class SquaredError(LossFunction):
    def value(self, y, t):
        c = tf.reduce_mean((y - t) ** 2, reduction_indices=[1])
        return tf.reduce_mean(c)


class MAE(LossFunction):
    def __init__(self, scale_fnc=lambda x: x):
        self.__scale_fnc = scale_fnc

    def value(self, y, t):
        c = tf.reduce_mean(tf.abs(self.__scale_fnc(y) - self.__scale_fnc(t)), reduction_indices=[1])
        return tf.reduce_mean(c)
