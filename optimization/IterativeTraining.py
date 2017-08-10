import datetime
from logging import Logger

import tensorflow as tf
import time
import os
import logging
from typing import List

from monitors.Criteria import Criterion, NullCriterion
from monitors.Quantity import FeedDict
from neuralflow.optimization.Algorithm import Algorithm


class IterativeTraining:
    def __init__(self, max_it: int, algorithm: Algorithm, output_dir: str, feed_dicts: List[FeedDict], logger: Logger):
        self.__max_it = max_it
        self.__algorithm = algorithm
        self.__output_dir = output_dir + "/"
        self.__log_filename = self.__output_dir + 'train.log'
        self.__feed_dicts = feed_dicts
        self.__stop_criterion = NullCriterion()
        self.__save_criterion = NullCriterion()
        self.__logger = logger

    def set_stop_criterion(self, criterion: Criterion):
        self.__stop_criterion = criterion

    def set_save_criterion(self, criterion: Criterion):
        self.__save_criterion = criterion

    def train(self, sess: tf.Session):
        self.__logger.info("Beginning training...")

        # sess = tf.Session()
        # self.__init_writers(sess)

        sess.run(tf.local_variables_initializer())  # TODO spostare?
        sess.run(tf.global_variables_initializer())

        stop = False
        i = 1

        start_time = time.time()
        t0 = time.time()
        while not stop:

            # train step
            train_step, train_dict = self.__algorithm.get_train_op()
            sess.run(train_step, feed_dict=train_dict)

            for d in self.__feed_dicts:
                d.feed(sess=sess, iteration=i)

            stop, _ = self.__stop_criterion.is_satisfied()
            if i == self.__max_it:
                self.__logger.info("Maximum number of iteration reached.")
                stop = True
            save, save_it = self.__save_criterion.is_satisfied()
            if save_it == i and save:
                tsave0 = time.time()
                self.__algorithm.save_check_point(output_dir=self.__output_dir, name="best_checkpoint", session=sess)
                tsave1 = time.time()
                self.__logger.info("Best model found -> checkpoint saved ({:.2f}s)".format(tsave1 - tsave0))

            if i % 100 == 0:
                t1 = time.time()
                self.__logger.info("Iteration: {}, time:{:.1f}s".format(i, t1 - t0))
                t0 = t1
            i += 1

        self.__logger.info("Done. (Training time:{:.1f}m)".format((time.time() - start_time) / 60))
