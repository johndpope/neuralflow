from typing import Dict, List
import tensorflow as tf

from optimization.OptimizationProblem import OptimizationProblem


class MultiTaskProblem(OptimizationProblem):
    def __init__(self, problems: List[OptimizationProblem]):
        self.__problems = problems
        self.__objective_fnc_value = 0
        for p in problems:
            self.__objective_fnc_value += p.objective_fnc_value
        self.__trainables = []
        for p in problems:
            self.__trainables += p.trainables
        self.__trainables = list(set(self.__trainables))  # remove duplicates

    def save_check_point(self, file: str, session: tf.Session):
        for i, p in enumerate(self.__problems):
            p.save_check_point("{}_i".format(file))

    @property
    def trainables(self):
        return self.__trainables

    @property
    def objective_fnc_value(self):
        return self.__objective_fnc_value

    def get_feed_dict(self) -> Dict:
        feed_dict = {}
        for p in self.__problems:
            feed_dict.update(p.get_feed_dict())
        return feed_dict
