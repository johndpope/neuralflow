import abc

import numpy as np
import tensorflow as tf
from neuralflow import ActivationFunction, TensorInitialization


class Layer:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def output(self, input):
        """return the symbolic output of the layer given the input"""

    @abc.abstractproperty
    def trainables(self):
        """returns the list of trainable variables"""

    @abc.abstractproperty
    def n_out(self):
        """returns the dimension of the output"""


class LayerProducer:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_layer(self, n_in: int, float_type, name: str):
        """return a layer which accepts an input of with dimension 'n_in' """


class RBFLayer(Layer):
    def __init__(self, U, beta, float_type, name: str):
        # matrix of centroids
        self.__U = tf.Variable(initial_value=U, name="{}_u".format(name), dtype=float_type)
        self.__beta = tf.Variable(initial_value=beta, name="{}_beta".format(name), dtype=float_type)
        self.__n_out = U.shape[0]

    @property
    def n_out(self):
        return self.__n_out

    def output(self, input):
        dist_matrix = tf.reduce_sum((tf.sub(input, self.__U)) ** 2, reduction_indices=2)
        dist_matrix = tf.transpose(dist_matrix, [1, 0])
        return self.__beta * tf.exp(-dist_matrix)

    @property
    def trainables(self):
        return [self.__U, self.__beta]


class RBFLayerProducer(LayerProducer):
    def __init__(self, n_units: int, initialization: TensorInitialization):
        self.__n_units = n_units
        self.__initialization = initialization

    def get_layer(self, n_in: int, float_type, name: str):
        U = self.__initialization.get(size=(self.__n_units, 1, n_in))
        beta = self.__initialization.get(size=(self.__n_units,))

        return RBFLayer(U=U, beta=beta, float_type=float_type, name=name)


class StandardLayer(Layer):
    def __init__(self, W, b, activation_fnc: ActivationFunction, float_type, name: str):
        self.__W = tf.Variable(initial_value=W, name=name + '_W', dtype=float_type)
        self.__b = tf.Variable(initial_value=b, name=name + '_b', dtype=float_type)
        self.__n_out = W.shape[1]
        self.__activation_fnc = activation_fnc

    @property
    def n_out(self):
        return self.__n_out

    def output(self, input):
        return self.__activation_fnc.apply(tf.matmul(input, self.__W) + self.__b)

    @property
    def trainables(self):
        return [self.__W, self.__b]


class StandardLayerProducer(LayerProducer):
    def __init__(self, n_units: int, initialization: TensorInitialization, activation_fnc: ActivationFunction):
        self.__n_units = n_units
        self.__initialization = initialization
        self.__activation_fnc = activation_fnc

    def get_layer(self, n_in: int, float_type, name: str = 'Unknown_Layer'):
        W = self.__initialization.get(size=(n_in, self.__n_units))
        # W = np.zeros(shape=(n_in, self.__n_units))
        b = np.zeros(shape=(self.__n_units,))  # TODO passare da fuori
        return StandardLayer(W, b, self.__activation_fnc, float_type, name)
