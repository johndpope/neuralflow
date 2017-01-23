import tensorflow as tf
from neuralflow.optimization.OptimizationProblem import OptimizationProblem
from neuralflow.optimization.OptimizationStep import OptimizationStep


class GradientDescent(OptimizationStep):
    def __init__(self, lr: float, max_norm: float = 1):
        self.__max_norm = max_norm
        self.__lr = lr

    def get_train_op(self, problem:OptimizationProblem):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.__lr)

        global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

        grads = tf.gradients(problem.objective_fnc_value, problem.trainables)
        clipped_grads, _ = tf.clip_by_global_norm(grads,
                                                  clip_norm=self.__max_norm)  # TODO learning rate strategy
        grad_var_pairs = zip(clipped_grads, problem.trainables)

        train_op = optimizer.apply_gradients(grad_var_pairs, global_step=global_step)

        grads = [tf.reshape(g, [-1]) for g in grads]
        gradient = tf.concat(0, grads)

        return train_op, gradient
