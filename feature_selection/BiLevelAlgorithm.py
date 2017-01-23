from typing import Dict

from optimization.Algorithm import Algorithm
from optimization.OptimizationProblem import OptimizationProblem
import tensorflow as tf
from optimization.OptimizationStep import OptimizationStep


class BiLevelAlgorithm(Algorithm):
    def __init__(self, leader: OptimizationProblem, follower: OptimizationProblem, optimization_step: OptimizationStep):
        self.__leader = leader
        self.__follower = follower

        self.__train_op_leader, self.__gradient_leader = optimization_step.get_train_op(self.__leader)
        self.__train_op_follower, self.__gradient_follower = optimization_step.get_train_op(self.__follower)

        self.__round = 0

    def save_check_point(self, output_dir: str, name: str, session: tf.Session):
        self.__leader.save_check_point(output_dir=output_dir, name=name, session=session)

    def get_train_op(self):
        # return self.__train_op_leader, self.__leader.get_feed_dict()

        self.__round += 1

        if self.__round == 10:
            self.__round = 0
            return self.__train_op_leader, self.__leader.get_feed_dict()
        else:
            return self.__train_op_follower, self.__follower.get_feed_dict()
