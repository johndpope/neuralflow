import abc
from typing import List

import numpy as np
import tensorflow as tf
from neuralflow.neuralnets.ActivationFunction import ActivationFunction, IdentityFunction, TanhActivationFunction

from neuralflow.TensorInitilization import TensorInitialization, GaussianInitialization
from neuralflow.optimization.Model import Model


class Layer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def output(self, input):
        """return the symbolic output of the layer given the input"""


class StandardLayer(Layer):
    def __init__(self, W, b, activation_fnc: ActivationFunction, float_type, name:str):
        self.__W = tf.Variable(initial_value=W, name=name + '_W', dtype=float_type)
        self.__b = tf.Variable(initial_value=b, name=name + '_b', dtype=float_type)
        self.__n_out = W.shape[1]
        self.__activation_fnc = activation_fnc

    @property
    def n_out(self):
        return self.__n_out

    def output(self, input):
        return self.__activation_fnc.apply(tf.matmul(input, self.__W) + self.__b)


class LayerProducer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_layer(self, n_in: int, float_type, name:str):
        """return a layer which accepts an input of with dimension 'n_in' """


class StandardLayerProducer(LayerProducer):
    def __init__(self, n_units: int, initialization: TensorInitialization, activation_fnc: ActivationFunction):
        self.__n_units = n_units
        self.__initialization = initialization
        self.__activation_fnc = activation_fnc

    def get_layer(self, n_in: int, float_type, name:str='Unknown_Layer'):
        W = self.__initialization.get(size=(n_in, self.__n_units))
        # W = np.zeros(shape=(n_in, self.__n_units))
        b = np.zeros(shape=(self.__n_units,))
        return StandardLayer(W, b, self.__activation_fnc, float_type, name)


class FeedForwardNeuralNet(Model):
    def __init__(self, n_in: int, layer_producers: List[LayerProducer], float_type=tf.float32):
        self.__float_type = float_type
        self.__n_in = n_in

        self.__layers = []
        self.__x_placeholder = tf.placeholder(dtype=float_type, shape=(None, n_in))

        next_input = self.__x_placeholder
        next_in_dim = self.__n_in
        name = "Layer_{}"
        count = 1
        for layer in layer_producers:
            l = layer.get_layer(n_in=next_in_dim, float_type=self.__float_type, name=name.format(count))
            self.__layers.append(l)
            next_input = l.output(next_input)
            next_in_dim = l.n_out
            count +=1

        self.__n_out = next_in_dim  # the number of output units is determined by the last layer
        self.__output = next_input

    @property
    def output(self):
        return self.__output

    @property
    def input(self):
        return self.__x_placeholder

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out


if __name__ == '__main__':
    print("Begin")
    hidden_layer_prod = StandardLayerProducer(n_units=50, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                              activation_fnc=TanhActivationFunction())
    output_layer_prod = StandardLayerProducer(n_units=1, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                              activation_fnc=IdentityFunction())
    net = FeedForwardNeuralNet(n_in=2, layer_producers=[hidden_layer_prod, hidden_layer_prod, output_layer_prod])
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    batch_size = 5
    feed_dict = {
        net.input: np.ones(shape=(batch_size, 2), dtype='float32') * 2
    }

    a = sess.run(net.output, feed_dict=feed_dict)
    print(a)

    sess.close()
