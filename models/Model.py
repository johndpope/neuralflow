import abc
import tensorflow as tf
import os


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

    def save(self, file: str, session: tf.Session):
        """save the model to file"""
        os.makedirs(os.path.dirname(file), exist_ok=True)
        saver = tf.train.Saver()
        saver.save(session, file)
        tf.add_to_collection("net.out", self.output)
        tf.add_to_collection("net.in", self.input)

        # Generates MetaGraphDef.
        saver.export_meta_graph(file + ".meta")


class ExternalInputModel(Model):
    def __init__(self, n_in: int, float_type="float32"):
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
