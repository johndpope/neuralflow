import abc


class Function:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def apply(self, x):
        """:param x: the input of the function
           :returns the symbolic output of the function computed on x"""

    @abc.abstractproperty
    def n_out(self):
        """:returns the dimensionality of the output"""

    @abc.abstractproperty
    def n_in(self):
        """:returns the dimensionality of the input"""

    @abc.abstractproperty
    def trainables(self)->list:
        """:returns a list of trainables vaiables"""
