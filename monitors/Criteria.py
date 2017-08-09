import abc
from typing import Tuple

from monitors.Quantity import Observer, AbstractScalarMonitor
import tensorflow as tf
import numpy as np


class Criterion(Observer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_satisfied(self) -> Tuple[bool, int]:
        """returns True or False whether the condition is satisfied or not and the iteration when this was detected"""

    @staticmethod
    def get_compare_fnc(direction: str, thr: float = 0):
        if direction == "<":
            return lambda x, y: x < y - thr, np.inf
        elif direction == ">":
            return lambda x, y: x > y + thr, -np.inf
        else:
            raise ValueError("unsupported direction {}. Available directions are '>' and '<'".format(direction))


class ThresholdCriterion(Criterion):
    def __init__(self, thr: float, direction: str, monitor: AbstractScalarMonitor):
        self.__thr = thr
        self.__compare_fnc, _ = Criterion.get_compare_fnc(direction=direction)
        self.__satisfied = False
        monitor.register(self)
        self.__it = 1

    def compute_and_update(self, new_value, sess: tf.Session, iteration: int, writer):
        self.__satisfied = self.__compare_fnc(new_value, self.__thr)
        self.__it = iteration

    def is_satisfied(self) -> Tuple[bool, int]:
        return self.__satisfied, self.__it


class MaxNoImproveCriterion(Criterion):
    def __init__(self, max_no_improve: int, direction: str, monitor: AbstractScalarMonitor, thr: float = 0):
        self.__max_no_improve = max_no_improve
        self.__compare_fnc, bound = Criterion.get_compare_fnc(direction=direction, thr=thr)
        self.__count = 0
        self.__last_value = bound
        self.__max_reached = False
        self.__it = 1
        monitor.register(self)

    def compute_and_update(self, new_value, sess: tf.Session, iteration: int, writer):

        improve_occured = self.__compare_fnc(new_value, self.__last_value)
        print("Max no improv called: {}".format(improve_occured))

        if improve_occured:
            self.__it = iteration
            self.__count = 0
            self.__last_value = new_value
        else:
            self.__count += 1

        self.__max_reached = self.__count > self.__max_no_improve

    def is_satisfied(self) -> Tuple[bool, int]:
        return self.__max_reached, self.__it


class ImprovedValueCriterion(Criterion):
    def __init__(self, direction: str, monitor: AbstractScalarMonitor):
        self.__compare_fnc, bound = Criterion.get_compare_fnc(direction=direction)
        self.__best_value = bound
        self.__improvement_occured = False
        monitor.register(self)
        self.__it = 1

    def compute_and_update(self, new_value, sess: tf.Session, iteration: int, writer):
        print("Improved values called: {}".format(self.__improvement_occured))
        if self.__compare_fnc(new_value, self.__best_value):
            self.__improvement_occured = True
            self.__best_value = new_value
            self.__it = iteration
        else:
            self.__improvement_occured = False

    def is_satisfied(self) -> Tuple[bool, int]:
        return self.__improvement_occured, self.__it


class NullCriterion(Criterion):
    def __init__(self):
        self.__always_false = False
        self.__it = 1

    def compute_and_update(self, data, sess: tf.Session, iteration: int, writer):
        self.__it = iteration

    def is_satisfied(self) -> Tuple[bool, int]:
        return self.__always_false, self.__it
