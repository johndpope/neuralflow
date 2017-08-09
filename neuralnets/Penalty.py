import tensorflow as tf
from math_utils import norm


class Penalty:
    def value(self, vars: list):
        vars_ = [tf.reshape(v, [-1]) for v in vars]
        v = tf.concat(0, vars_)
        return norm(v)
