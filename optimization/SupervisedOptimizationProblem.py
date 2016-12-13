import tensorflow as tf
from neuralflow.models.Model import Model
from neuralflow import BatchProducer
from neuralflow.neuralnets.LossFunction import LossFunction
from neuralflow.optimization.OptimizationProblem import OptimizationProblem


class SupervisedOptimizationProblem(OptimizationProblem):
    def __init__(self, model: Model, loss_fnc: LossFunction, batch_producer: BatchProducer, batch_size: int):
        self.__model = model
        self.__loss_fnc = loss_fnc
        self.__batch_producer = batch_producer
        self.__t = tf.placeholder(tf.float32, shape=[None, self.__model.n_out], name="labels")  # labels
        self.__batch_size = batch_size

    @property
    def objective_fnc_value(self):
        return self.__loss_fnc.value(self.__model.output, self.__t)

    def get_feed_dict(self):
        batch = self.__batch_producer.get_batch(batch_size=self.__batch_size)
        feed_dict = {self.__model.input: batch["input"], self.__t: batch["output"]}
        return feed_dict

    @property
    def labels(self):
        return self.__t

    @property
    def input(self):
        return self.__model.input

    @property
    def trainables(self):
        return self.__model.trainables

    def save_check_point(self, file: str, session: tf.Session, id: int = 0):
        self.__model.save(file, session, id)
