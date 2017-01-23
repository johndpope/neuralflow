import abc

from neuralflow.optimization import OptimizationProblem
from neuralflow.optimization.OptimizationStep import OptimizationStep
import tensorflow as tf


class Algorithm:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_train_op(self):
        """:returns a train operation to run at each step"""

    @abc.abstractmethod
    def save_check_point(self, output_dir: str, name:str, session: tf.Session):
        """saves the models"""


class SimpleAlgorithm(Algorithm):
    def __init__(self, problem: OptimizationProblem, optimization_step: OptimizationStep):
        self.__problem = problem
        self.__step = optimization_step

        self.__train_op, self.__gradient = optimization_step.get_train_op(self.__problem)

    def get_train_op(self):
        return self.__train_op, self.__problem.get_feed_dict()

    @property
    def gradient(self):
        return self.__gradient

    def save_check_point(self, output_dir: str, name:str, session: tf.Session):
        self.__problem.save_check_point(output_dir=output_dir, name=name, session=session)

