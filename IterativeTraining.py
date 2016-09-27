from typing import List

import tensorflow as tf
from neuralflow.StoppingCriterion import StoppingCriterion
from neuralflow import BatchProducer
from neuralflow import Model
from neuralflow import ObjectiveFunction
from neuralflow import ValidationProducer

from .Monitor import Monitor


class IterativeTraining(object):
    def __init__(self, max_it: int, batch_size: int, model: Model, batch_producer: BatchProducer,
                 validation_producer: ValidationProducer, optimizer,
                 obj_fnc: ObjectiveFunction):
        self.__max_it = max_it
        self.__batch_size = batch_size
        self.__optimizer = optimizer
        self.__batch_producer = batch_producer
        self.__validation_producer = validation_producer
        self.__model = model
        self.__obj_fnc = obj_fnc

        self.__t = tf.placeholder(tf.float32, shape=[None, self.__model.n_out], name="labels")  # labels
        self.__loss = self.__obj_fnc.value(y=self.__model.output, t=self.__t)  # loss

        self.__monitor_dict = {}
        self.__stopping_criteria = []

    @property
    def labels(self):
        return self.__t

    @property
    def loss(self):
        return self.__loss

    def set_monitors(self, name: str, monitors: List[Monitor], feed_dict: dict, freq: int):
        assert (freq > 0)
        self.__monitor_dict[freq] = {
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

    def train(self):
        print("Beginning training...")

        # train step
        train_step = self.__optimizer.minimize(self.__loss)

        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())

        self.__init_writers(sess)

        stop = False
        i = 0

        while not stop:
            print(i)

            batch = self.__batch_producer.get_batch(batch_size=self.__batch_size)
            train_dict = {self.__model.input: batch["input"], self.__t: batch["output"]}
            train_step.run(feed_dict=train_dict)

            for f in self.__monitor_dict.keys():
                if i % f == 0:
                    m = self.__monitor_dict[f]
                    output = sess.run(m["ops"] + [m["summary"]],
                                      feed_dict=train_dict if m["feed_dict"] is None else m["feed_dict"])
                    summary = output[-1]
                    m["writer"].add_summary(summary, i)
            i += 1
            if i == self.__max_it or self.__stopping_criteria_satisfied():
                print("Stopping criterion satisfied")
                stop = True

        print("Done.")

        sess.close()
