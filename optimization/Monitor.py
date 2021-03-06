import abc

import tensorflow as tf
from neuralflow.utils.auc import auc


class Monitor(object):
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def summary(self):
        """return a tf summary object"""

    @property
    @abc.abstractmethod
    def value(self) -> float:
        """return the last observed value for the monitored variable"""

    @property
    @abc.abstractmethod
    def update_op(self):
        """returns an operation that should be run to update the monitor. Could be None"""


class ScalarMonitor(Monitor):
    def __init__(self, variable, name: str):
        self.__variable = variable
        self.__summary = tf.summary.scalar(name, variable)
        self.__stored_value = tf.Variable(0., dtype='float64', name='ScalarMonitor')

    @property
    def update_op(self):
        return self.__stored_value.assign(tf.cast(self.__variable, dtype='float64'))

    @property
    def value(self) -> float:
        return tf.cast(self.__variable, dtype='float64')

    @property
    def summary(self):
        return self.__summary


class AccuracyMonitor(Monitor): # FIXME
    def __init__(self, predictions, labels):
        self.__accuracy, self.__update_op = tf.contrib.metrics.streaming_accuracy(predictions, labels,
                                                                                  weights=None,
                                                                                  metrics_collections=None,
                                                                                  updates_collections=None,
                                                                                  name="accuracy_monitor")

        self.__summary = tf.summary.scalar('accuracy', self.__accuracy)

    @property
    def value(self) -> float:
        return self.__accuracy

    @property
    def summary(self):
        return self.__summary

    @property
    def update_op(self):
        return self.__update_op


class RocMonitor(Monitor):
    def __init__(self, predictions, labels):
        # self.__auc_score, self.__auc_update_op = tf.contrib.metrics.streaming_auc(predictions=predictions,
        #                                                                           labels=labels,
        #                                                                           num_thresholds=200,
        #                                                                           metrics_collections=None,
        #                                                                           updates_collections=None,
        #                                                                           name="auc_monitor")

        self.__auc_score = auc(predictions=predictions, labels=labels, num_thresholds=200, metrics_collections=None,
                               name="auc_monitor")

        self.__summary = tf.scalar_summary('auc_score', self.__auc_score)

    @property
    def update_op(self):
        # return self.__auc_update_op
        return []

    @property
    def summary(self):
        return self.__summary

    @property
    def value(self):
        return tf.cast(self.__auc_score, dtype=tf.float64)
