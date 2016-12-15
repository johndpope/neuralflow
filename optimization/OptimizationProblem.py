import abc
from typing import Dict
import tensorflow as tf


class FeedDictionaryProducer:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_feed_dict(self) -> Dict:
        """returns a dictionary of inputs placeholder:values"""


class OptimizationProblem(FeedDictionaryProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def objective_fnc_value(self):
        """"""

    @abc.abstractproperty
    def trainables(self):
        """"""

    @abc.abstractmethod
    def save_check_point(self, output_dir: str, name:str, session: tf.Session):
        """"""
