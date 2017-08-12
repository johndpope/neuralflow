import abc

import tensorflow as tf
import numpy as np


class LossFunction:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def value(self, y, t):
        """returns the (symbolic) value of the objective function given a value 'y' and the targets 't'"""


class CrossEntropy(LossFunction):
    def __init__(self, single_output: bool = False, class_weights: np.ndarray = None):
        self.__single_output = single_output
        self.__class_weights = class_weights if class_weights is not None else np.array((1, 1))  # TODO class !=2

    def value(self, y, t):
        if self.__single_output:
            c = -tf.reduce_sum((self.__class_weights[1] * t * tf.log(y + 1e-10) +
                                (1 - t) * self.__class_weights[0] * tf.log((1 - y) + 1e-10)), reduction_indices=[1])
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


class HingeLoss(LossFunction):
    def value(self, y, t):
        t_ = t * 2 - 1
        y_ = y * 2 - 1
        c = tf.reduce_mean(tf.maximum(0., 1. - t_ * y_), reduction_indices=[1])
        return tf.reduce_mean(c)


class EpsilonInsensitiveLoss(LossFunction):
    def __init__(self, epsilon:float=0.1, scale_fnc=lambda x: x):
        self.__scale_fnc = scale_fnc
        self.__epsilon = epsilon

    def value(self, y, t):
        c = tf.abs(self.__scale_fnc(y) - self.__scale_fnc(t)) - self.__epsilon
        c = tf.maximum(0., c)
        c = tf.reduce_mean(c, reduction_indices=[1])
        return tf.reduce_mean(c)
