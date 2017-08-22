from logging import Logger

import tensorflow as tf
import time
from typing import List, Dict

from monitors import CheckPointer
from monitors.Criteria import Criterion, NullCriterion
from monitors.Quantity import ExternalFeed, Feed
from neuralflow.optimization.Algorithm import Algorithm


class IterativeFeed(Feed):  # fixme duplicate code
    def __init__(self, freq: int, output_dir: str, logger:Logger):
        self.__freq = freq
        self.__quantities = []
        self.__output_dir = output_dir
        self.__name = "TrainBatch"
        self.__writer = None
        self.__logger = logger

    def add_quantity(self, quantity):
        self.__quantities.append(quantity)

    def feed(self, sess: tf.Session, iteration: int, train_op, feed_dict: Dict):
        """executes the train operation alongside the registered quantities and start the notify process"""
        if not self.__writer:
            self.__writer = tf.summary.FileWriter(self.__output_dir + self.__name, sess.graph)

        if iteration % self.__freq == 0:
            t0 = time.time()
            run_list = []
            for q in self.__quantities:
                run_list.append(q.tf_quantity)

            output = sess.run([train_op] + run_list, feed_dict=feed_dict)
            for o, q in zip(output[1:], self.__quantities):
                event_dict = {"iteration": iteration, "source_name": self.__name, "writer": self.__writer,
                              "updated_value": o}
                q.compute_and_update(sess, event_dict=event_dict)
            t1 = time.time()
            self.__logger.info("Iteration: {}, time:{:.1f}s".format(iteration, t1 - t0))
        else:
            sess.run(train_op, feed_dict=feed_dict)


class IterativeTraining:
    def __init__(self, max_it: int, algorithm: Algorithm, output_dir: str, feed_dicts: List[ExternalFeed],
                 logger: Logger,
                 check_pointer: CheckPointer = None):
        self.__max_it = max_it
        self.__algorithm = algorithm
        self.__output_dir = output_dir + "/"
        self.__log_filename = self.__output_dir + 'train.log'
        self.__feed_dicts = feed_dicts
        self.__stop_criterion = NullCriterion()
        self.__save_criterion = NullCriterion()
        self.__logger = logger
        self.__check_pointer = check_pointer
        self.__iterative_feed = IterativeFeed(100, output_dir, logger)

    @property
    def iterative_feed(self) -> IterativeFeed:
        return self.__iterative_feed

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
            self.__iterative_feed.feed(sess, i, train_op=train_step, feed_dict=train_dict)

            for d in self.__feed_dicts:
                d.feed(sess=sess, iteration=i)

            stop, _ = self.__stop_criterion.is_satisfied()
            if not stop and i == self.__max_it:
                self.__logger.info("Maximum number of iteration reached.")
                stop = True

            i += 1

        self.__logger.info("Done. (Training time:{:.1f}m)".format((time.time() - start_time) / 60))
