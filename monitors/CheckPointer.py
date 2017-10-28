from logging import Logger

from neuralflow.monitors.Criteria import Criterion
from neuralflow.monitors.Quantity import Observer
import tensorflow as tf
import time
import os

from neuralflow.optimization.Algorithm import Algorithm


class CheckPointer(Observer):
    def __init__(self, output_dir: str, save_criterion: Criterion, algorithm: Algorithm, logger: Logger):
        self.__output_dir = output_dir
        self.__logger = logger
        save_criterion.register(self)
        self.__algorithm = algorithm
        os.makedirs(output_dir, exist_ok=True)

    def compute_and_update(self, sess: tf.Session, event_dict: dict):
        save = event_dict["updated_value"]
        if save:
            t0_save = time.time()
            self.__algorithm.save_check_point(output_dir=self.__output_dir, name="best_checkpoint", session=sess)
            t1_save = time.time()
            self.__logger.info("Best model found -> checkpoint saved ({:.2f}s)".format(t1_save - t0_save))
