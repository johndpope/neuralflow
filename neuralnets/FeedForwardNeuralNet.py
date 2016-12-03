from typing import List

import numpy as np
import tensorflow as tf
from neuralflow.TensorInitilization import GaussianInitialization
from neuralflow.models.Model import Model, ExternalInputModel
from neuralflow.neuralnets.ActivationFunction import IdentityFunction, TanhActivationFunction
from neuralnets.Layers import LayerProducer, StandardLayerProducer


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
