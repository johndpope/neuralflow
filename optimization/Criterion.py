import abc
from typing import List

from neuralflow.optimization.Monitor import Monitor
import tensorflow as tf
import numpy as np


class Criterion(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_satisfied(self) -> List:
        """returns True or False whether the condition is satisfied or not """  # TODO fix description

    @staticmethod
    def get_compare_fnc(direction: str):
        if direction == "<":
            return lambda x, y: x < y, np.inf
        elif direction == ">":
            return lambda x, y: x > y, -np.inf
        else:
            raise ValueError("unsupported direction {}. Available directions are '>' and '<'".format(direction))


class ThresholdCriterion(Criterion):
    def __init__(self, monitor: Monitor, thr: float, direction: str):
        self.__monitor = monitor
        self.__thr = thr
        self.__compare_fnc, _ = Criterion.get_compare_fnc(direction=direction)

    def is_satisfied(self) -> List:
        return [self.__compare_fnc(self.__monitor.value, self.__thr)]


class MaxNoImproveCriterion(Criterion):
    def __init__(self, max_no_improve: int, monitor: Monitor, direction: str):
        self.__monitor = monitor
        self.__max_no_improve = max_no_improve
        self.__count = tf.Variable(initial_value=0, name='max_no_improve_count', dtype=tf.int32)
        self.__compare_fnc, bound = Criterion.get_compare_fnc(direction=direction)
        self.__last_value = tf.Variable(initial_value=bound, name='max_no_improve_last_value', dtype=tf.float64)

        improve_occured = self.__compare_fnc(tf.cast(self.__monitor.value, dtype=tf.float64), self.__last_value)
        self.__count_op = self.__count.assign(
            tf.cond(improve_occured, lambda: tf.constant(0), lambda: tf.add(self.__count, 1)))
        self.__last_value_op = self.__last_value.assign(
            tf.cond(improve_occured, lambda: tf.cast(self.__monitor.value, dtype=tf.float64), lambda: self.__last_value,
                    name="max_no_improve_last_value_op"))
        self.__max_reached = self.__count > self.__max_no_improve

    def is_satisfied(self) -> List:
        return [self.__count_op, self.__last_value_op, self.__max_reached]


class ImprovedValueCriterion(Criterion):
    def __init__(self, monitor: Monitor, direction: str):
        self.__monitor = monitor
        self.__compare_fnc, bound = Criterion.get_compare_fnc(direction=direction)
        self.__best_value = tf.Variable(initial_value=bound, name='improved_value_criterion_best_last_value',
                                        dtype=tf.float64)

        self.__improve_occured = self.__compare_fnc(self.__monitor.value, self.__best_value)
        self.__best_value_op = self.__best_value.assign(
            tf.cond(self.__improve_occured, lambda: self.__monitor.value, lambda: self.__best_value,
                    name="improved_value_criterion_best_last_value_op"))

    def is_satisfied(self) -> List:
        return [self.__best_value_op, self.__improve_occured]


class NullCriterion(Criterion):
    def __init__(self):
        self.__always_false = tf.constant(False, dtype=tf.bool)

    def is_satisfied(self) -> List:
        return [self.__always_false]
