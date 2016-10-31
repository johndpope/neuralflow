import abc

from neuralflow.optimization.Monitor import Monitor
import tensorflow as tf


class StoppingCriterion(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_satisfied(self) -> bool:
        """returns True or False whether the condition is satisfied or not """

    @staticmethod
    def get_compare_fnc(direction: str):
        if direction == "<":
            return lambda x, y: x < y
        elif direction == ">":
            return lambda x, y: x > y
        else:
            raise ValueError("unsupported direction {}. Available directions are '>' and '<'".format(direction))


class ThresholdCriterion(StoppingCriterion):
    def __init__(self, monitor: Monitor, thr: float, direction: str):
        self.__monitor = monitor
        self.__thr = thr
        self.__compare_fnc = StoppingCriterion.get_compare_fnc(direction=direction)

    def is_satisfied(self):
        return [self.__compare_fnc(self.__monitor.value, self.__thr)]  # TODO passare sessione qui?


class MaxNoImproveCriterion(StoppingCriterion):
    def __init__(self, max_no_improve: int, monitor: Monitor, direction: str):
        self.__monitor = monitor
        self.__max_no_improve = max_no_improve
        self.__count = tf.Variable(initial_value=0, name='max_no_improve_count', dtype=tf.int32)
        self.__compare_fnc = StoppingCriterion.get_compare_fnc(direction=direction)
        self.__last_value = tf.Variable(initial_value=0, name='max_no_improve_last_value', dtype=tf.float32)

        improve_occured = self.__compare_fnc(self.__monitor.value, self.__last_value)
        self.__count_op = self.__count.assign(tf.cond(improve_occured, lambda: tf.constant(0), lambda: tf.add(self.__count, 1)))
        self.__last_value_op = self.__last_value.assign(
            tf.cond(improve_occured, lambda: self.__monitor.value, lambda: self.__last_value,
                    name="max_no_improve_last_value_op"))
        self.__max_reached = self.__count > self.__max_no_improve

    def is_satisfied(self) -> bool:
        return [self.__count_op, self.__last_value_op, self.__max_reached]
