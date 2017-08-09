import abc

import numpy as np
import tensorflow as tf
from neuralflow.TensorInitilization import TensorInitialization
from neuralflow.neuralnets.ActivationFunction import ActivationFunction


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
        dist_matrix = tf.reduce_sum((tf.subtract(input, self.__U)) ** 2, reduction_indices=2)
        dist_matrix = tf.transpose(dist_matrix, [1, 0])
        return tf.exp(-dist_matrix * self.__beta)

    @property
    def trainables(self):
        return [self.__U, self.__beta]


class RBFLayerProducer(LayerProducer):
    def __init__(self, n_units: int, initialization: TensorInitialization):
        self.__n_units = n_units
        self.__initialization = initialization

    def get_layer(self, n_in: int, float_type, name: str):
        U = self.__initialization.get(size=(self.__n_units, 1, n_in))
        # beta = self.__initialization.get(size=(self.__n_units,))
        beta = np.ones(shape=(self.__n_units,))

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


class ElementwiseMulLayer(Layer):
    def __init__(self, beta, float_type, name):
        self.__beta = tf.Variable(initial_value=beta, name=name + '_beta', dtype=float_type)
        self.__n_out = beta.shape[0]

    @property
    def n_out(self):
        return self.__n_out

    def output(self, input):
        return self.__beta * input

    @property
    def trainables(self):
        return [self.__beta]


class ElementwiseMulLayerProducer(LayerProducer):
    def __init__(self, initialization: TensorInitialization):
        self.__initialization = initialization

    def get_layer(self, n_in: int, float_type, name: str = 'Unknown_Layer'):
        beta = self.__initialization.get(size=(n_in,))
        return ElementwiseMulLayer(beta, float_type, name)


if __name__ == "__main__":
    n_units = 5
    n_in = 3

    U_np = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
    U_np = np.reshape(U_np, newshape=(n_units, 1, n_in))
    beta_np = np.array([1, 2, 3, 4, 5])
    x_np = np.array([[1, 1, 1], [2, 2, 2]])

    print("beta.shape: ", beta_np.shape)
    print("U.shape: ", U_np.shape)
    print("x.shape", x_np.shape)

    x = tf.placeholder(dtype=tf.float32, shape=(None, n_in), name="ExternalInput")
    U = tf.Variable(initial_value=U_np, name="U", dtype=tf.float32)
    beta = tf.Variable(initial_value=beta_np, name="beta", dtype=tf.float32)

    n_out = U_np.shape[0]

    s = (tf.sub(x, U)) ** 2
    dist_matrix = tf.reduce_sum(s, reduction_indices=2)
    dist_matrix = tf.transpose(dist_matrix, [1, 0])

    eb = dist_matrix * beta
    e = tf.exp(-dist_matrix * beta)

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    result = sess.run(s, feed_dict={x: x_np})
    print(result)
    print(result.shape)

    print("====")
    print(U_np)
    print("=====")
    print(x_np)

    print("?????")
    result = sess.run(e, feed_dict={x: x_np})
    print(result)
    print(result.shape)
