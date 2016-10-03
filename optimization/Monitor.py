import abc

import tensorflow as tf


class Monitor(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def summary(self):
        """return a tf summary object"""

    @abc.abstractproperty
    def value(self)->float:
        """return the last observed value for the monitored variable"""

    @abc.abstractproperty
    def update_op(self):
        """returns an operation that should be run to update the monitor. Could be None"""


class ScalarMonitor(Monitor):

    def __init__(self, variable, name:str):
        self.__variable = variable
        self.__summary = tf.scalar_summary(name, variable)
        self.__stored_value = tf.Variable(0., dtype='float64', name='ScalarMonitor')

    @property
    def update_op(self):
        return self.__stored_value.assign(tf.cast(self.__variable, dtype='float64'))

    @property
    def value(self)->float:
        return self.__stored_value.eval()

    @property
    def summary(self):
        return  self.__summary


class RocMonitor(Monitor):
    @property
    def update_op(self):
        return self.__auc_update_op

    @property
    def summary(self):
        return self.__summary

    @property
    def value(self):
        return self.__auc_score

    def __init__(self, prediction, labels):
        self.__auc_score, self.__auc_update_op = tf.contrib.metrics.streaming_auc(predictions=prediction, labels=labels,
                                                                                  ignore_mask=None, num_thresholds=200,
                                                                                  metrics_collections=None,
                                                                                  updates_collections=None,
                                                                                  name="auc_score")
        self.__summary = tf.scalar_summary('auc_score', self.__auc_score)
