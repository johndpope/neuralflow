from typing import List

import tensorflow as tf
from neuralflow.SessionManager import SessionManager
from neuralflow.optimization.Monitor import Monitor
from neuralflow.optimization.OptimizationProblem import OptimizationProblem
from neuralflow.optimization.StoppingCriterion import StoppingCriterion
from neuralflow.optimization.Optimizer import Optimizer


class IterativeTraining(object):
    def __init__(self, max_it: int, optimizer: Optimizer, problem: OptimizationProblem):
        self.__max_it = max_it
        self.__optimizer = optimizer
        self.__problem = problem

        self.__monitor_dict = {}
        self.__stopping_criteria = []

    def add_monitors(self, name: str, monitors: List[Monitor], freq: int, feed_dict: dict = None):
        assert (freq > 0)
        self.__monitor_dict[str(freq) + str(name)] = {
            "summary": tf.merge_summary([m.summary for m in monitors], name="merged_summary_" + name),
            "feed_dict": feed_dict,
            "freq": freq,
            "name": name,
            "ops": [m.update_op for m in monitors if m.update_op is not None]
        }

    def set_stopping_criterion(self, criteria: List[StoppingCriterion]):
        self.__stopping_criteria = criteria  # or like

    def __stopping_criteria_satisfied(self):
        result = False
        for criterion in self.__stopping_criteria:
            result = result or criterion.is_satisfied()
        return result

    def __init_writers(self, sess):
        out_dir = "/home/giulio/tensorBoards/"
        for m in self.__monitor_dict.values():
            m["writer"] = tf.train.SummaryWriter(out_dir + m["name"], sess.graph)

    def train(self, sess):
        print("Beginning training...")

        sess.run(tf.initialize_all_variables())  # TODO spostare?
        # train step
        train_step = self.__optimizer.train_op

        self.__init_writers(sess)

        stop = False
        i = 0

        while not stop:
            if i % 100 == 0:
                print("Iteration: {}".format(i))

            train_dict = self.__problem.get_feed_dict()
            # train_step.run(feed_dict=train_dict)
            sess.run(train_step, feed_dict=train_dict)

            for id in self.__monitor_dict.keys():
                m = self.__monitor_dict[id]
                f = m["freq"]
                if i % f == 0:
                    output = sess.run(m["ops"] + [m["summary"]],
                                      feed_dict=train_dict if m["feed_dict"] is None else m["feed_dict"])
                    summary = output[-1]
                    m["writer"].add_summary(summary, i)
            i += 1
            if i == self.__max_it or sess.run(self.__stopping_criteria_satisfied()):
                print("Stopping criterion satisfied")
                stop = True

        print("Done.")
