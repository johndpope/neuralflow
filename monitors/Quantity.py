import abc
from typing import Dict

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score


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
                q.compute_and_update(o, sess, iteration, self.__writer)


class Observer:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute_and_update(self, data, sess: tf.Session, iteration: int, writer):
        """updates the monitor with the received data"""


class Quantity:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute_and_update(self, data, sess: tf.Session, iteration: int, writer):
        """updated the quantity and notifies every registered observer of the happened change"""

    @abc.abstractmethod
    def register(self, o: Observer):
        """adds 'Observer' o to the list of registered objects"""


class PrimitiveQuantity(Quantity):
    def __init__(self, quantity, name: str):
        self.__quantity = quantity
        self.__monitors = []
        self.__name = name

    @property
    def tf_quantity(self):
        return self.__quantity

    def register(self, o: Observer):
        self.__monitors.append(o)

    def compute_and_update(self, data, sess: tf.Session, iteration: int, writer):
        print("Received update for primitive quantity: {}".format(self.__name))
        for m in self.__monitors:
            m.compute_and_update(data, sess, iteration, writer)


class AbstractScalarMonitor(Observer, Quantity):
    __metaclass__ = abc.ABCMeta


class ScalarMonitor2(AbstractScalarMonitor):
    def __init__(self, name: str):
        self.__value_np = None
        self.__name = name
        self.__monitors = []

        self.__value_tf = tf.Variable(0, name=name + "_monitor")
        self.__summary = tf.summary.scalar(name, self.__value_tf)

    def register(self, o: Observer):
        self.__monitors.append(o)

    def compute_and_update(self, new_value_np, sess: tf.Session, iteration: int, writer):
        self.__value_np = new_value_np
        print("Updating Scalar ({}) Monitor -> value: {:.2f}".format(self.__name, new_value_np))

        assign_op = self.__value_tf.assign(self.__value_np)
        summary, _ = sess.run([self.__summary, assign_op])
        writer.add_summary(summary, iteration)
        for m in self.__monitors:
            m.compute_and_update(new_value_np, sess, iteration, writer)


class AccuracyMonitor2(AbstractScalarMonitor):
    def __init__(self, labels):
        self.__labels_np = np.argmax(labels, axis=1)
        self.__scala_monitor = ScalarMonitor2(name="accuracy")

    def compute_and_update(self, y_np, sess: tf.Session, iteration: int, writer):
        pred_classes = np.argmax(y_np, axis=1)
        acc_np = accuracy_score(self.__labels_np, pred_classes)
        self.__scala_monitor.compute_and_update(acc_np, sess, iteration, writer)

    def register(self, o: Observer):
        return self.__scala_monitor.register(o)
