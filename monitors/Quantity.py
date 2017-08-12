import abc
from logging import Logger
from typing import Dict

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score


def capitalize_first(s: str):
    return s[0].upper() + s[1:]


def updated_event_dict(old_dict: dict, new_name: str = None, new_value=None):
    event_dict = old_dict.copy()
    if new_name is not None:
        source_name = "{}@{} ".format(capitalize_first(new_name), capitalize_first(event_dict["source_name"]))
        event_dict["source_name"] = source_name
    if new_value is not None:
        event_dict["updated_value"] = new_value
    return event_dict


def print_event(event_dict: dict, value_format: str = "") -> str:
    s = "{} -> {" + value_format + "}"
    return s.format(event_dict["source_name"], event_dict["updated_value"])


class FeedDict:
    def __init__(self, feed_dict: Dict, freq: int, output_dir: str, name: str):
        self.__feed_dict = feed_dict
        self.__freq = freq
        self.__quantities = []
        self.__output_dir = output_dir
        self.__name = name

        self.__writer = None

    def add_quantity(self, quantity):
        self.__quantities.append(quantity)

    def feed(self, sess: tf.Session, iteration: int):
        if not self.__writer:
            self.__writer = tf.summary.FileWriter(self.__output_dir + self.__name, sess.graph)

        if iteration % self.__freq == 0:
            run_list = []
            for q in self.__quantities:
                run_list.append(q.tf_quantity)

            output = sess.run(run_list, feed_dict=self.__feed_dict)
            for o, q in zip(output, self.__quantities):
                event_dict = {"iteration": iteration, "source_name": self.__name, "writer": self.__writer,
                              "updated_value": o}
                q.compute_and_update(sess, event_dict=event_dict)

    @property
    def name(self):
        return self.__name


class Observer:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute_and_update(self, sess: tf.Session, event_dict: dict):
        """updates itself and notifies every registered observer of the happened change"""


class Quantity(Observer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def register(self, o: Observer):
        """adds 'Observer' o to the list of registered objects"""


class QuantityImpl:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def update_dict(self, sess: tf.Session, event_dict: dict):
        """receive an updated dict and computes its update"""


class ConcreteQuantity:
    def __init__(self, quantity_impl: QuantityImpl):
        self.__monitors = []
        self.__impl = quantity_impl

    def register(self, o: Observer):
        self.__monitors.append(o)

    def compute_and_update(self, sess: tf.Session, event_dict: dict):
        event_dict = self.__impl.update_dict(sess, event_dict)
        for m in self.__monitors:
            m.compute_and_update(sess, event_dict)


class PrimitiveQuantity(Quantity, QuantityImpl):
    def __init__(self, quantity, name: str, feed: FeedDict):
        self.__quantity = quantity
        self.__monitors = []
        self.__name = name
        self.__feed = feed
        self.__feed.add_quantity(self)
        self.__impl = ConcreteQuantity(self)

    @property
    def tf_quantity(self):
        return self.__quantity

    def update_dict(self, sess, event_dict):
        return event_dict

    def register(self, o: Observer):
        return self.__impl.register(o)

    def compute_and_update(self, sess: tf.Session, event_dict: dict):
        self.__impl.compute_and_update(sess, event_dict)


class AbstractScalarMonitor(Quantity):
    __metaclass__ = abc.ABCMeta


class ScalarMonitorImpl(QuantityImpl):
    def __init__(self, name: str, logger: Logger):
        self.__value_np = None
        self.__name = name
        self.__impl = QuantityImpl()

        self.__value_tf = tf.Variable(0, name=name + "_monitor")
        self.__summary = tf.summary.scalar(name, self.__value_tf)
        self.__logger = logger

    def update_dict(self, sess: tf.Session, event_dict: dict):
        self.__value_np = event_dict["updated_value"]

        assign_op = self.__value_tf.assign(self.__value_np)
        summary, _ = sess.run([self.__summary, assign_op])
        event_dict["writer"].add_summary(summary, event_dict["iteration"])

        event_dict = updated_event_dict(new_name=self.__name, old_dict=event_dict)
        if self.__logger:
            self.__logger.info(print_event(event_dict, value_format=":.2f"))
        return event_dict


class ScalarMonitor(AbstractScalarMonitor):
    def __init__(self, name: str, logger: Logger):
        self.__quantity = ConcreteQuantity(ScalarMonitorImpl(name, logger))

    def register(self, o: Observer):
        self.__quantity.register(o)

    def compute_and_update(self, sess: tf.Session, event_dict: dict):
        self.__quantity.compute_and_update(sess, event_dict)


class AccuracyMonitorImpl(QuantityImpl):
    def __init__(self, labels, logger: Logger):
        self.__labels_np = labels
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            self.__labels_np = np.argmax(labels, axis=1)
        self.__scalar_monitor = ScalarMonitorImpl(name="accuracy", logger=logger)

    def update_dict(self, sess: tf.Session, event_dict: dict):
        pred_classes = event_dict["updated_value"]
        if len(pred_classes.shape) > 1 and pred_classes.shape[1] > 1:
            pred_classes = np.argmax(pred_classes, axis=1)
        else:
            pred_classes = np.round(pred_classes)

        acc_np = accuracy_score(self.__labels_np, pred_classes)

        event_dict = updated_event_dict(old_dict=event_dict, new_value=acc_np)
        return self.__scalar_monitor.update_dict(sess, event_dict)


class AccuracyMonitor(AbstractScalarMonitor):
    def __init__(self, labels, logger: Logger):
        self.__quantity = ConcreteQuantity(AccuracyMonitorImpl(labels, logger))

    def register(self, o: Observer):
        self.__quantity.register(o)

    def compute_and_update(self, sess: tf.Session, event_dict: dict):
        self.__quantity.compute_and_update(sess, event_dict)
