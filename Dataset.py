import abc
from abc import abstractmethod


class BatchProducer(object):
    __metaclass__ = abc.ABCMeta

    @abstractmethod
    def get_batch(self, batch_size):
        """return a batch used for training. A batch is a dict {input, output}. Input is a numpy array of size
        (batch_size, n_inputs), output is of size (batch_size, n_outputs)"""


class ValidationProducer(object):
    __metaclass__ = abc.ABCMeta

    @abstractmethod
    def get_validation(self):
        """return a batch to be used as a validation set. A batch is a dict {input, output}. Input is a numpy array of size
        (batch_size, n_inputs), output is of size (batch_size, n_outputs)"""



