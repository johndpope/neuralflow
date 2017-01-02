from queue import Queue
from threading import Thread, Lock
from typing import List

import progressbar
from enel.svm.Model import TrainingStrategy

lock = Lock()


class ThreadProgressBar:
    def __init__(self, max_value):
        self.__n = 0
        self.__max_value = max_value
        self.__progress = progressbar.ProgressBar(max_value=max_value)
        self.__progress.update(self.__n)

    def update(self):
        self.__n += 1
        self.__progress.update(self.__n)

        # Only for a good formatted output (ProgressBar computes window's width in the wrong way...)
        if self.__n == self.__max_value:
            print()


class ModelEvaluation:
    def __init__(self, method, n_threads: int = 1):
        assert (n_threads >= 1)
        self.__n_threads = n_threads  # TODO max threads
        self.__method = method

    def run(self, strategies: List[TrainingStrategy]) -> dict:
        """:returns the model with highest score computed using a evaluation method"""

        all_configurations = []
        if self.__n_threads == 1:

            progress_bar = ThreadProgressBar(max_value=len(strategies))
            for strategy in strategies:
                predictions, score = self.__method.evaluate(strategy)
                all_configurations.append({"strategy": strategy, "predictions": predictions, "score": score})
                progress_bar.update()
        else:
            queue = Queue(maxsize=self.__n_threads)
            threads = []
            progress_bar = ThreadProgressBar(max_value=len(strategies))
            for strategy in strategies:
                threads.append(TrainThread(validation_method=self.__method, strategy=strategy, queue=queue,
                                           progress_bar=progress_bar))

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            for thread in threads:
                # print("Configuration:", thread.results['model'], "score:", thread.results["score"])
                all_configurations.append({"model": thread.results['model'], "score": thread.results["score"],
                                           "predictions": thread.results['predictions']})

        return all_configurations


class TrainThread(Thread):
    def __init__(self, validation_method, strategy: TrainingStrategy, queue: Queue, progress_bar: ThreadProgressBar):
        Thread.__init__(self)
        self.__validation_method = validation_method
        self.__strategy = strategy
        self.__results = None
        self.__queue = queue
        self.__progress_bar = progress_bar

    def run(self):
        self.__queue.put(True, block=True)
        predictions, score = self.__validation_method.evaluate(self.__strategy)
        self.__results = {
            "predictions": predictions,
            "model": self.__strategy,
            "score": score
        }
        self.__queue.get(block=True)

        with lock:
            self.__progress_bar.update()

    @property
    def results(self):
        return self.__results
