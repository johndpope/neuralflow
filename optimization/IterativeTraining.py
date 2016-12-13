import datetime
import tensorflow as tf
import time
import os
import logging
from typing import List
from neuralflow.optimization.Monitor import Monitor
from neuralflow.optimization.OptimizationProblem import OptimizationProblem
from neuralflow.optimization.Criterion import Criterion, NullCriterion
from neuralflow.optimization.Optimizer import Optimizer


class IterativeTraining(object):
    def __init__(self, max_it: int, optimizer: Optimizer, problem: OptimizationProblem, output_dir: str):
        self.__max_it = max_it
        self.__optimizer = optimizer
        self.__problem = problem

        self.__monitor_dict = {}
        self.__output_dir = output_dir + "/"
        self.__log_filename = self.__output_dir + 'train.log'

    def add_monitors_and_criteria(self, name: str, freq: int, monitors: List[Monitor],
                                  saving_criteria: List[Criterion] = (),
                                  stopping_criteria: List[Criterion] = (), feed_dict: dict = None):
        assert (freq > 0)
        self.__monitor_dict[str(freq) + str(name)] = {
            "summary": tf.merge_summary([m.summary for m in monitors], name="merged_summary_" + name),
            "feed_dict": feed_dict,
            "freq": freq,
            "name": name,
            "ops": [m.update_op for m in monitors if m.update_op is not None],
            "saving_criteria": [NullCriterion().is_satisfied()] if len(saving_criteria) == 0 else  [c.is_satisfied() for
                                                                                                    c in
                                                                                                    saving_criteria],
            "stopping_criteria": [NullCriterion().is_satisfied()] if len(stopping_criteria) == 0 else [c.is_satisfied()
                                                                                                       for c in
                                                                                                       stopping_criteria]
        }

    # def set_saving_criterion(self, criteria: List[Criterion]):
    #     self.__saving_criteria = criteria  # or like
    #
    # def set_stopping_criterion(self, criteria: List[Criterion]):
    #     self.__stopping_criteria = criteria  # or like

    @staticmethod
    def __criteria_satisfied(criteria_results):
        """performs or of all criteria"""
        result = False
        for r in criteria_results:
            result = result or r[-1]
        return result

    def __init_writers(self, sess):
        for m in self.__monitor_dict.values():
            m["writer"] = tf.train.SummaryWriter(self.__output_dir + m["name"], sess.graph)

    def __start_logger(self):
        os.makedirs(self.__output_dir, exist_ok=True)

        file_handler = logging.FileHandler(filename=self.__log_filename, mode='a')
        formatter = logging.Formatter('%(levelname)s:%(message)s')
        file_handler.setFormatter(formatter)

        logger = logging.getLogger('rnn.train' + self.__output_dir)
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

        # train step
        train_step = self.__optimizer.train_op

        # sess = tf.Session()

        self.__init_writers(sess)

        sess.run(tf.initialize_all_variables())  # TODO spostare?
        sess.run(tf.initialize_local_variables())

        stop = False
        i = 1

        start_time = time.time()
        t0 = time.time()
        while not stop:

            train_dict = self.__problem.get_feed_dict()
            sess.run(train_step, feed_dict=train_dict)

            save = False
            for id in self.__monitor_dict.keys():
                m = self.__monitor_dict[id]
                f = m["freq"]
                if i % f == 0:
                    run_list = [m["summary"], m["ops"], m["saving_criteria"], m["stopping_criteria"]]
                    output = sess.run(run_list,
                                      feed_dict=train_dict if m["feed_dict"] is None else m["feed_dict"])
                    # output[-1]
                    summary, save_crit_res, stop_crit_res = output[0], output[2], output[3]
                    if m["name"] == "validation": logger.info("{} {}".format(m["name"], output[2][0][0]))
                    m["writer"].add_summary(summary, i)
                    if IterativeTraining.__criteria_satisfied(save_crit_res):
                        save = True

                    if i == self.__max_it or IterativeTraining.__criteria_satisfied(stop_crit_res):
                        logger.info("Stopping criterion satisfied")
                        stop = True
            if save:
                tsave0 = time.time()
                self.__problem.save_check_point(file=self.__output_dir + "best_checkpoint", session=sess)
                tsave1 = time.time()
                logger.info("Best model found -> checkpoint saved ({:.2f}s)".format(tsave1 - tsave0))

            if i % 100 == 0:
                t1 = time.time()
                logger.info("Iteration: {}, time:{:.1f}s".format(i, t1 - t0))
                t0 = t1
            i += 1

        logger.info("Done. (Training time:{:.1f}m)".format((time.time() - start_time) / 60))
