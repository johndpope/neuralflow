import abc
from typing import List
import tensorflow as tf
from neuralflow.math_utils import merge_all_tensors
from neuralflow.models import Model
from neuralflow.monitors.Quantity import Observer, AbstractScalarMonitor
import numpy as np


class Penalty:
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def value_tf(self):
        """:returns a tensor symbol of the penalty"""


class NormPenalty(Penalty):
    def __init__(self, quantities_tf: List):
        # self.__value_tf = 0
        # for q in quantities_tf:
        #     self.__value_tf += tf.norm(q, 2)
        #
        # print(quantities_tf[1])
        # self.__value_tf = norm(quantities_tf[0], "l2")

        vars = [tf.reshape(g, [-1]) for g in quantities_tf]
        v = tf.concat(vars, 0)  # FIXME

        self.__value_tf = tf.reduce_mean(tf.abs(v))

    @property
    def value_tf(self):
        return self.__value_tf


class ValidationErrorPenalty(Penalty, Observer):
    @property
    def value_tf(self):
        return self.__penalty_tf

    def compute_and_update(self, sess: tf.Session, event_dict: dict):
        updated_value = event_dict["updated_value"]

        if updated_value > self.__best_value:
            self.__best_value = updated_value
            self.__n_it_worse = 0
        else:
            if self.__n_it_worse <= self.__freq:
                self.__n_it_worse += 1
            else:
                # self.__weights_sum += updated_value
                weight = 1-updated_value
                assign_op = self.__centroid.assign_add(weight * self.__merged)
                add_op = self.__weights_sum.assign_add(weight)
                sess.run([assign_op, add_op])
                self.__n_acc += 1
                self.__n_it_worse = 0
                self.__best_value = 0
                out = sess.run([self.__centroid, self.__weights_sum, self.__penalty_tf])
                print("acc: {:.2f}".format(updated_value))
                print("norm_c: {:.2e}, weight_sum {:.2f}, penalty: {:.2e}".format(np.linalg.norm(out[0]), out[1], out[2]))
                print("appended new centroid: total={}".format(self.__n_acc))

    def __init__(self, model: Model, monitor: AbstractScalarMonitor, freq: int):
        monitor.register(self)
        self.__model = model
        self.__freq = freq

        self.__merged = merge_all_tensors(self.__model.trainables)

        shape = (self.__merged.get_shape().as_list()[0],)
        self.__centroid = tf.Variable(initial_value=np.zeros(shape=shape), name="x_centroid", dtype="float32")  # FIXME float32
        self.__weights_sum = tf.Variable(initial_value=1e-7, name="weight_sum", dtype="float32")  #FIXME
        self.__n_acc = 0
        self.__best_value = 0  # FIXME
        self.__n_it_worse = 0

        argument = self.__weights_sum * tf.norm(self.__merged) ** 2 - 2 * tf.reduce_sum(
            tf.multiply(self.__merged, self.__centroid))
        self.__penalty_tf = tf.exp(-tf.norm(argument))
