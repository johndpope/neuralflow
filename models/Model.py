import abc
import tensorflow as tf

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


class ExternalInputModel(Model):

    def __init__(self, n_in:int, float_type="float32"):
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



