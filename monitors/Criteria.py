import abc
from logging import Logger
from typing import Tuple

from neuralflow.monitors.Quantity import Observer, AbstractScalarMonitor, Quantity, QuantityImpl, updated_event_dict, print_event, \
    ConcreteQuantity
import tensorflow as tf
import numpy as np


class Criterion:
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


class ThresholdCriterion(Criterion, Quantity):
    def compute_and_update(self, sess: tf.Session, event_dict: dict):
        return self.__quantity.compute_and_update(sess, event_dict)

    def register(self, o: Observer):
        return self.__quantity.register(o)

    def is_satisfied(self) -> Tuple[bool, int]:
        return self.__impl.is_satisfied()

    def __init__(self, thr: float, direction: str, monitor: AbstractScalarMonitor, logger: Logger = None):
        monitor.register(self)
        self.__impl = ThresholdCriterionImpl(thr, direction, logger)
        self.__quantity = ConcreteQuantity(self.__impl)


class ThresholdCriterionImpl(Criterion, QuantityImpl):
    def __init__(self, thr: float, direction: str, logger: Logger = None):
        self.__thr = thr
        self.__compare_fnc, _ = Criterion.get_compare_fnc(direction=direction)
        self.__satisfied = False
        self.__it = 1
        self.__logger = logger

    def update_dict(self, sess: tf.Session, event_dict: dict):
        self.__satisfied = self.__compare_fnc(event_dict["updated_value"], self.__thr)
        self.__it = event_dict["iteration"]

        event_dict = updated_event_dict(old_dict=event_dict, new_value=self.__satisfied, new_name="ThresholdCriterion")
        if self.is_satisfied() and self.__logger:
            self.__logger.info(print_event(event_dict))

        return event_dict

    def is_satisfied(self) -> Tuple[bool, int]:
        return self.__satisfied, self.__it


class MaxNoImproveCriterion(Criterion, Quantity):
    def compute_and_update(self, sess: tf.Session, event_dict: dict):
        return self.__quantity.compute_and_update(sess, event_dict)

    def register(self, o: Observer):
        return self.__quantity.register(o)

    def is_satisfied(self) -> Tuple[bool, int]:
        return self.__impl.is_satisfied()

    def __init__(self, max_no_improve: int, direction: str, monitor: AbstractScalarMonitor, thr: float = 0,
                 logger: Logger = None):
        monitor.register(self)
        self.__impl = MaxNoImproveCriterionImpl(max_no_improve, direction, thr, logger)
        self.__quantity = ConcreteQuantity(self.__impl)


class MaxNoImproveCriterionImpl(Criterion, QuantityImpl):
    def __init__(self, max_no_improve: int, direction: str, thr: float = 0,
                 logger: Logger = None):
        self.__max_no_improve = max_no_improve
        self.__compare_fnc, bound = Criterion.get_compare_fnc(direction=direction, thr=thr)
        self.__count = 0
        self.__last_value = bound
        self.__max_reached = False
        self.__it = 1
        self.__logger = logger

    def update_dict(self, sess: tf.Session, event_dict: dict):

        new_value = event_dict["updated_value"]
        improve_occurred = self.__compare_fnc(new_value, self.__last_value)

        if improve_occurred:
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

        return event_dict

    def is_satisfied(self) -> Tuple[bool, int]:
        return self.__max_reached, self.__it


class ImprovedValueCriterion(Criterion, Quantity):
    def compute_and_update(self, sess: tf.Session, event_dict: dict):
        return self.__quantity.compute_and_update(sess, event_dict)

    def register(self, o: Observer):
        return self.__quantity.register(o)

    def is_satisfied(self) -> Tuple[bool, int]:
        return self.__impl.is_satisfied()

    def __init__(self, direction: str, monitor: AbstractScalarMonitor, logger: Logger = None):
        monitor.register(self)
        self.__impl = ImprovedValueCriterionImpl(direction, logger)
        self.__quantity = ConcreteQuantity(self.__impl)


class ImprovedValueCriterionImpl(Criterion, QuantityImpl):
    def __init__(self, direction: str, logger: Logger = None):
        self.__compare_fnc, bound = Criterion.get_compare_fnc(direction=direction)
        self.__best_value = bound
        self.__improvement_occured = False
        self.__it = 1
        self.__logger = logger

    def update_dict(self, sess: tf.Session, event_dict: dict):
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

        return event_dict

    def is_satisfied(self) -> Tuple[bool, int]:
        return self.__improvement_occured, self.__it


class NullCriterion(Criterion, Quantity):
    def compute_and_update(self, sess: tf.Session, event_dict: dict):
        return self.__quantity.compute_and_update(sess, event_dict)

    def register(self, o: Observer):
        return self.__quantity.register(o)

    def is_satisfied(self) -> Tuple[bool, int]:
        return self.__impl.is_satisfied()

    def __init__(self):
        self.__impl = NullCriterionImpl()
        self.__quantity = ConcreteQuantity(self.__impl)


class NullCriterionImpl(Criterion, QuantityImpl):
    def __init__(self):
        self.__always_false = False
        self.__it = 1

    def update_dict(self, sess: tf.Session, event_dict: dict):
        self.__it = event_dict["iteration"]
        event_dict = updated_event_dict(old_dict=event_dict, new_value=False,
                                        new_name="NullCriterion")
        return event_dict

    def is_satisfied(self) -> Tuple[bool, int]:
        return self.__always_false, self.__it
