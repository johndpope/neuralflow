from typing import List
import tensorflow as tf
from neuralflow.models.Function import Function
from neuralflow.neuralnets.Layers import LayerProducer


class FeedForwardNeuralNet(Function):
    layer_name_proto = "{}_Layer_{}"

    def __init__(self, n_in, layer_producers: List[LayerProducer] = (), float_type=tf.float32, name: str = "UnNamed_NN"):
        super().__init__()
        assert len(layer_producers) >= 0
        self.__float_type = float_type
        self.__n_in = n_in
        self.__n_out = n_in

        self.__layers = []
        self.__trainables = []
        self.__name = name

        for layer_prod in layer_producers:
            self.add_layer(layer_prod)

    def add_layer(self, layer_producer: LayerProducer):
        l = layer_producer.get_layer(n_in=self.__n_out, float_type=self.__float_type,
                                     name=FeedForwardNeuralNet.layer_name_proto.format(self.__name,
                                                                                       len(self.__layers) + 1))
        self.__trainables += l.trainables
        self.__layers.append(l)
        self.__n_out = l.n_out
        return self

    def apply(self, x):
        output = x
        for l in self.__layers:
            output = l.output(output)
        return output

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out

    @property
    def trainables(self):
        return self.__trainables
