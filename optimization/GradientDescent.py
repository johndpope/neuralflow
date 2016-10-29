import tensorflow as tf
from neuralflow.optimization.OptimizationProblem import OptimizationProblem
from neuralflow.optimization.Optimizer import Optimizer


class GradientDescent(Optimizer):
    def __init__(self, problem: OptimizationProblem, lr: float, max_norm: float = 1):
        self.__optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.__max_norm = max_norm
        self.__lr = lr
        self.__global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')
        self.__grads = tf.gradients(problem.objective_fnc_value, problem.trainables)
        clipped_grads, _ = tf.clip_by_global_norm(self.__grads,
                                                  clip_norm=self.__max_norm)  # TODO learning rate strategy
        grad_var_pairs = zip(clipped_grads, problem.trainables)

        self.__train_op = self.__optimizer.apply_gradients(grad_var_pairs, global_step=self.__global_step)

    @property
    def train_op(self):
        return self.__train_op

    @property
    def gradient(self):
        grads = [tf.reshape(g, [-1]) for g in self.__grads]

        return tf.concat(0, grads)
