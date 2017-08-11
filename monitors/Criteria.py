import abc
from logging import Logger
from typing import Tuple

from monitors.Quantity import Observer, AbstractScalarMonitor, Quantity, QuantityImpl, updated_event_dict, print_event
import tensorflow as tf
import numpy as np


class Criterion(Quantity):
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
    def __init__(self, thr: float, direction: str, monitor: AbstractScalarMonitor, logger: Logger = None):
        self.__thr = thr
        self.__compare_fnc, _ = Criterion.get_compare_fnc(direction=direction)
        self.__satisfied = False
        monitor.register(self)
        self.__it = 1
        self.__logger = logger
        self.__quantity_impl = QuantityImpl()

    def register(self, o: Observer):
        self.__quantity_impl.register(o)

    def compute_and_update(self, sess: tf.Session, event_dict: dict):
        self.__satisfied = self.__compare_fnc(event_dict["updated_value"], self.__thr)
        self.__it = event_dict["iteration"]

        event_dict = updated_event_dict(old_dict=event_dict, new_value=self.__satisfied, new_name="ThresholdCriterion")
        if self.is_satisfied() and self.__logger:
            self.__logger.info(print_event(event_dict))

        self.__quantity_impl.compute_and_update(sess, event_dict)

    def is_satisfied(self) -> Tuple[bool, int]:
        return self.__satisfied, self.__it


class MaxNoImproveCriterion(Criterion):
    def __init__(self, max_no_improve: int, direction: str, monitor: AbstractScalarMonitor, thr: float = 0,
                 logger: Logger = None):
        self.__max_no_improve = max_no_improve
        self.__compare_fnc, bound = Criterion.get_compare_fnc(direction=direction, thr=thr)
        self.__count = 0
        self.__last_value = bound
        self.__max_reached = False
        self.__it = 1
        self.__logger = logger
        monitor.register(self)

        self.__quantity_impl = QuantityImpl()

    def register(self, o: Observer):
        self.__quantity_impl.register(o)

    def compute_and_update(self, sess: tf.Session, event_dict: dict):

        new_value = event_dict["updated_value"]
        improve_occured = self.__compare_fnc(new_value, self.__last_value)

        if improve_occured:
            self.__it = event_dict["iteration"]
            self.__count = 0
            self.__last_value = new_value

        else:
            self.__count += 1

        self.__max_reached = self.__count > self.__max_no_improve

        event_dict = updated_event_dict(old_dict=event_dict, new_value=self.__max_reached,
                                        new_name="MaxNoImproveCriterion")
        if self.__max_reached and self.__logger:
            self.__logger.info(print_event(event_dict))

        self.__quantity_impl.compute_and_update(sess, event_dict)

    def is_satisfied(self) -> Tuple[bool, int]:
        return self.__max_reached, self.__it


class ImprovedValueCriterion(Criterion):
    def __init__(self, direction: str, monitor: AbstractScalarMonitor, logger: Logger = None):
        self.__compare_fnc, bound = Criterion.get_compare_fnc(direction=direction)
        self.__best_value = bound
        self.__improvement_occured = False
        monitor.register(self)
        self.__it = 1
        self.__logger = logger

        self.__quantity_impl = QuantityImpl()

    def register(self, o: Observer):
        self.__quantity_impl.register(o)

    def compute_and_update(self, sess: tf.Session, event_dict: dict):
        new_value = event_dict["updated_value"]

        if self.__compare_fnc(new_value, self.__best_value):
            self.__improvement_occured = True
            self.__best_value = new_value
            self.__it = event_dict["iteration"]
        else:
            self.__improvement_occured = False

        event_dict = updated_event_dict(old_dict=event_dict, new_value=self.__improvement_occured,
                                        new_name="ImprovedValueCriterion")
        if self.__improvement_occured and self.__logger:
            self.__logger.info(print_event(event_dict))

        self.__quantity_impl.compute_and_update(sess, event_dict)

    def is_satisfied(self) -> Tuple[bool, int]:
        return self.__improvement_occured, self.__it


class NullCriterion(Criterion):
    def __init__(self):
        self.__always_false = False
        self.__it = 1

    def register(self, o: Observer):
        pass

    def compute_and_update(self, sess: tf.Session, event_dict: dict):
        self.__it = event_dict["iteration"]

    def is_satisfied(self) -> Tuple[bool, int]:
        return self.__always_false, self.__it
