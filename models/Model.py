import abc
import tensorflow as tf
import os

import time


class Model(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def input(self):
        """returns the placeholder for the inputs of the model"""

    @abc.abstractproperty
    def output(self):
        """returns the placeholder for the outputs of the model"""

    @abc.abstractproperty
    def n_in(self):
        """returns the dimension of the input"""

    @abc.abstractproperty
    def n_out(self):
        """returns the dimension of the output"""

    @abc.abstractproperty
    def trainables(self):
        """returns the list of trainable variables"""

    def __init__(self):
        self.__meta_graph_saved = False

    def save(self, file: str, session: tf.Session):  # FIXME
        """save the model to file"""
        self.__saver = tf.train.Saver(var_list=self.trainables)

        self.__saver.save(session, file, write_meta_graph=False)
        if not self.__meta_graph_saved:
            tf.add_to_collection("net.out", self.output)
            tf.add_to_collection("net.in", self.input)
            os.makedirs(os.path.dirname(file), exist_ok=True)
            # Generates MetaGraphDef.
            self.__saver.export_meta_graph(file + ".meta")
            self.__meta_graph_saved = True


class ExternalInputModel(Model):
    def __init__(self, n_in: int, float_type="float32"):
        super().__init__()
        self.__n_in = n_in
        self.__input_placeholder = tf.placeholder(dtype=float_type, shape=(None, n_in), name="ExternalInput")

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_in

    @property
    def output(self):
        return self.__input_placeholder

    @property
    def input(self):
        return self.__input_placeholder

    @property
    def trainables(self):
        return []
