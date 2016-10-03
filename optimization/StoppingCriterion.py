import abc

from neuralflow.optimization.Monitor import Monitor


class StoppingCriterion(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_satisfied(self) -> bool:
        """returns True or False whether the condition is satisfied or not """


class ThresholdCriterion(StoppingCriterion):
    def __init__(self, monitor: Monitor, thr: float, direction: str):
        self.__monitor = monitor
        self.__thr = thr

        if direction == "<":
            self.__compare_fnc = lambda x, y: x < y
        elif direction == ">":
            self.__condition = lambda x, y: x > y
        else:
            raise ValueError("unsupported direction {}. Available directions are '>' and '<'".format(direction))

    def is_satisfied(self):
        return self.__compare_fnc(self.__monitor.value, self.__thr)
