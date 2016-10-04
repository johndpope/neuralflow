import abc

import tensorflow as tf


class ActivationFunction(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def apply(self, x: tf.Tensor):
        """apply the function to 'x' and return the computed value"""


class TanhActivationFunction(ActivationFunction):
    def apply(self, x: tf.Tensor):
        return tf.tanh(x)


class IdentityFunction(ActivationFunction):
    def apply(self, x: tf.Tensor):
        return x


class SoftmaxActivationFunction(ActivationFunction):

    def __init__(self, single_output: bool = False):
        self.__single_output = single_output

    def apply(self, x: tf.Tensor):
        if self.__single_output:
            return tf.reduce_mean(1. / (1. + tf.exp(x)), reduction_indices=0)  # x should be 1-dimensional
        else:
            return tf.nn.softmax(x)
