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


class HingeLoss(LossFunction):
    def value(self, y, t):
        t_ = t * 2 - 1
        y_ = y * 2 - 1
        c = tf.reduce_mean(tf.maximum(0., 1. - t_ * y_), reduction_indices=[1])
        return tf.reduce_mean(c)


class EpsilonInsensitiveLoss(LossFunction):
    def __init__(self, epsilon: float = 0.1, scale_fnc=lambda x: x):
        self.__scale_fnc = scale_fnc
        self.__epsilon = epsilon

    def value(self, y, t):
        c = tf.abs(self.__scale_fnc(y) - self.__scale_fnc(t)) - self.__epsilon
        c = tf.maximum(0., c)
        c = tf.reduce_mean(c, reduction_indices=[1])
        return tf.reduce_mean(c)


# class MAE(LossFunction):
#     def __init__(self, scale_fnc=lambda x: x):
#         self.__scale_fnc = scale_fnc
#
#     def value(self, y, t):
#         c = tf.reduce_mean(tf.abs(self.__scale_fnc(y) - self.__scale_fnc(t)), reduction_indices=[1])
#         return tf.reduce_mean(c)


class MAE(LossFunction):
    """Implementation of mean absolute error as a loss function. Supports nans in the target"""

    def __init__(self, scale_fnc=lambda x: x):
        self.__scale_fnc = scale_fnc

    def value(self, y, t):
        """
        :param y: matrix of the predictions. Must be a (n_samples, n_features) matrix.
        :param t: matrix of the targets. Must be a (n_samples, n_features) matrix. Can contain nans (which gets ignored).
        :returns the mean absolute error between mean(|y-t|)
        """
        cleaned = tf.where(tf.is_nan(t), tf.zeros_like(t), tf.abs(self.__scale_fnc(y) - self.__scale_fnc(t)))

        c = tf.reduce_mean(cleaned, reduction_indices=[1])
        return tf.reduce_mean(c)


class PearsonLoss(LossFunction):
    """Implementation of the pearson correlation as a loss function. Supports nans in the targets."""

    def value(self, y, t):
        """
        :param y: matrix of the predictions. Must be a (n_samples, n_features) matrix.
        :param t: matrix of the targets. Must be a (n_samples, n_features) matrix. Can contain nans (which gets ignored).
        :returns the negative mean pearson correlation between the columns of y and t
        """
        not_nans = tf.where(tf.is_nan(t), tf.zeros_like(t), tf.ones_like(t))
        n = tf.reduce_sum(not_nans, reduction_indices=[0])

        y_ = tf.where(tf.is_nan(t), tf.zeros_like(t), y)
        t_ = tf.where(tf.is_nan(t), tf.zeros_like(t), t)

        y_ = y_ - tf.reduce_sum(y_, reduction_indices=[0]) / n
        t_ = t_ - tf.reduce_sum(t_, reduction_indices=[0]) / n

        num = tf.reduce_sum(y_ * t_, reduction_indices=[0])
        den = tf.sqrt(tf.reduce_sum(y_ ** 2, reduction_indices=[0])) * tf.sqrt(
            tf.reduce_sum(t_ ** 2, reduction_indices=[0]))
        corr = num / den
        mean_corr = tf.reduce_mean(corr)
        return -mean_corr
