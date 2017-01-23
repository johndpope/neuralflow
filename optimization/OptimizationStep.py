import abc

from neuralflow.optimization.OptimizationProblem import OptimizationProblem


class OptimizationStep:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_train_op(self, problem: OptimizationProblem):
        """returns an operation that can be run in a session that
        execute a training step and the gradient symbolic variable"""
