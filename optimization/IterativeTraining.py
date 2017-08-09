import datetime
import tensorflow as tf
import time
import os
import logging
from typing import List

from monitors.Criteria import Criterion, NullCriterion
from monitors.Quantity import FeedDict
from neuralflow.optimization.Algorithm import Algorithm


class IterativeTraining:
    def __init__(self, max_it: int, algorithm: Algorithm, output_dir: str, feed_dicts: List[FeedDict]):
        self.__max_it = max_it
        self.__algorithm = algorithm
        self.__output_dir = output_dir + "/"
        self.__log_filename = self.__output_dir + 'train.log'
        self.__feed_dicts = feed_dicts
        self.__stop_crit = NullCriterion()
        self.__save_crit = NullCriterion()

    def set_stop_criterion(self, criterion: Criterion):
        self.__stop_crit = criterion

    def set_save_criterion(self, criterion: Criterion):
        self.__save_crit = criterion

    #
    # def add_monitors_and_criteria(self, name: str, freq: int, monitors: List[Monitor],
    #                               saving_criteria: List[Criterion] = (),
    #                               stopping_criteria: List[Criterion] = (), feed_dict: dict = None):
    #     assert (freq > 0)
    #     self.__monitor_dict[str(freq) + str(name)] = {
    #         "summary": tf.summary.merge([m.summary for m in monitors], name="merged_summary_" + name),
    #         "feed_dict": feed_dict,
    #         "freq": freq,
    #         "name": name,
    #         "ops": [m.update_op for m in monitors if m.update_op is not None],
    #         "saving_criteria": [NullCriterion().is_satisfied()] if len(saving_criteria) == 0 else  [c.is_satisfied() for
    #                                                                                                 c in
    #                                                                                                 saving_criteria],
    #         "stopping_criteria": [NullCriterion().is_satisfied()] if len(stopping_criteria) == 0 else [c.is_satisfied()
    #                                                                                                    for c in
    #                                                                                                    stopping_criteria]
    #     }
    #
    # @staticmethod
    # def __criteria_satisfied(criteria_results):
    #     """performs or of all criteria"""
    #     result = False
    #     for r in criteria_results:
    #         result = result or r[-1]
    #     return result

    # def __init_writers(self, sess):
    #     for m in self.__monitor_dict.values():
    #         m["writer"] = tf.summary.FileWriter(self.__output_dir + m["name"], sess.graph)

    def __start_logger(self):
        os.makedirs(self.__output_dir, exist_ok=True)

        file_handler = logging.FileHandler(filename=self.__log_filename, mode='a')
        formatter = logging.Formatter('%(levelname)s:%(message)s')
        file_handler.setFormatter(formatter)

        logger = logging.getLogger('sgd_train' + self.__output_dir)
        logger.setLevel(logging.INFO)

        for hdlr in logger.handlers:  # remove all old handlers
            logger.removeHandler(hdlr)
        logger.addHandler(file_handler)  # set the new handler
        now = datetime.datetime.now()

        logger.info('starting logging activity in date {}'.format(now.strftime("%d-%m-%Y %H:%M")))
        return logger

    def train(self, sess: tf.Session):
        logger = self.__start_logger()
        logger.info("Beginning training...")

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

            stop, _ = self.__stop_crit.is_satisfied()
            if i == self.__max_it:
                logger.info("Maximum number of iteration reached.")
                stop = True
            save, save_it = self.__save_crit.is_satisfied()
            if save_it == i and save:
                tsave0 = time.time()
                self.__algorithm.save_check_point(output_dir=self.__output_dir, name="best_checkpoint", session=sess)
                tsave1 = time.time()
                logger.info("Best model found -> checkpoint saved ({:.2f}s)".format(tsave1 - tsave0))

            if i % 100 == 0:
                t1 = time.time()
                logger.info("Iteration: {}, time:{:.1f}s".format(i, t1 - t0))
                t0 = t1
            i += 1

        logger.info("Done. (Training time:{:.1f}m)".format((time.time() - start_time) / 60))
