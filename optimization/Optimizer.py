import abc


class Optimizer:
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def train_op(self):
        """returns an operation that can be run in a session that execute a training step"""


