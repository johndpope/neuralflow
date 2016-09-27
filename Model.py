import abc


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
