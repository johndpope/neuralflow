import tensorflow as tf
from neuralflow.Dataset import BatchProducer
from neuralflow.models.Model import Model
from neuralflow.neuralnets.LossFunction import LossFunction
from neuralflow.optimization.OptimizationProblem import OptimizationProblem


class SupervisedOptimizationProblem(OptimizationProblem):
    def __init__(self, model: Model, loss_fnc: LossFunction, batch_producer: BatchProducer, batch_size: int, trainables:list=None, penalty:float=0):  # XXX batch producer
        self.__model = model
        self.__loss_fnc = loss_fnc
        self.__batch_producer = batch_producer
        self.__t = tf.placeholder(tf.float32, shape=[None, self.__model.n_out], name="labels")  # labels
        self.__batch_size = batch_size
        self.__trainables = trainables if trainables is not None and len(trainables) > 0 else model.trainables

        self.__lambda = penalty if penalty is not None else 0

    @property
    def objective_fnc_value(self):
        vars = [tf.reshape(g, [-1]) for g in self.__trainables]
        v = tf.concat(0, vars)

        return self.__loss_fnc.value(self.__model.output, self.__t) + self.__lambda * tf.reduce_mean(tf.abs(v))

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
        return self.__trainables

    def save_check_point(self, output_dir: str, name:str, session: tf.Session):
        self.__model.save(output_dir, name, session)
