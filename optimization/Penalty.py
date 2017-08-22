import abc
from typing import List
import tensorflow as tf
from math_utils import norm


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



