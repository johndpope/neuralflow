import abc
from typing import List

import numpy as np
import tensorflow as tf
from neuralflow.models.Model import Model, ExternalInputModel
from neuralflow.TensorInitilization import TensorInitialization, GaussianInitialization
from neuralflow.neuralnets.ActivationFunction import ActivationFunction, IdentityFunction, TanhActivationFunction


class Layer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def output(self, input):
        """return the symbolic output of the layer given the input"""

    @abc.abstractproperty
    def trainables(self):
        """returns the list of trainable variables"""


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


class LayerProducer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_layer(self, n_in: int, float_type, name: str):
        """return a layer which accepts an input of with dimension 'n_in' """


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


class FeedForwardNeuralNet(Model):
    layer_name_proto = "Layer_{}"

    def __init__(self, input_model: Model, layer_producers: List[LayerProducer] = (), float_type=tf.float32):
        super().__init__()
        assert len(layer_producers) >= 0
        self.__float_type = float_type
        self.__input_model = input_model
        self.__n_in = self.__input_model.n_in

        self.__layers = []
        self.__trainables = []
        # self.__x_placeholder = input_producer.input

        # next_input = self.__input_model.output
        # next_in_dim = self.__input_model.n_out
        # count = 1
        # for layer in layer_producers:
        #     l = layer.get_layer(n_in=next_in_dim, float_type=self.__float_type,
        #                         name=FeedForwardNeuralNet.layer_name_proto.format(count))
        #     self.__trainables += l.trainables
        #     self.__layers.append(l)
        #     next_input = l.output(next_input)
        #     next_in_dim = l.n_out
        #     count += 1
        #
        # self.__n_out = next_in_dim  # the number of output units is determined by the last layer
        # self.__output = next_input

        self.__n_out = self.__input_model.n_out
        self.__output = self.__input_model.output
        for layer_prod in layer_producers:
            self.add_layer(layer_prod)

    def add_layer(self, layer_producer: LayerProducer):
        l = layer_producer.get_layer(n_in=self.__n_out, float_type=self.__float_type,
                                     name=FeedForwardNeuralNet.layer_name_proto.format(len(self.__layers) + 1))
        self.__trainables += l.trainables
        self.__layers.append(l)
        self.__output = l.output(self.__output)
        self.__n_out = l.n_out

    @property
    def output(self):
        return self.__output

    @property
    def input(self):
        return self.__input_model.input

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out

    @property
    def trainables(self):
        return self.__trainables


if __name__ == '__main__':
    print("Begin")
    hidden_layer_prod = StandardLayerProducer(n_units=50, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                              activation_fnc=TanhActivationFunction())
    output_layer_prod = StandardLayerProducer(n_units=1, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                              activation_fnc=IdentityFunction())
    net = FeedForwardNeuralNet(ExternalInputModel(n_in=2),
                               layer_producers=[hidden_layer_prod, hidden_layer_prod, output_layer_prod])
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    batch_size = 5
    feed_dict = {
        net.input: np.ones(shape=(batch_size, 2), dtype='float32') * 2
    }

    a = sess.run(net.output, feed_dict=feed_dict)
    print(a)

    sess.close()
