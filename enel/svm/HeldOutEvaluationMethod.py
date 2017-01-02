from enel.svm.Model import TrainingStrategy


class HeldOutEvaluationMethod:
    def __init__(self, train_set, validation_set, score_fnc):
        self.__train_set = train_set
        self.__validation_set = validation_set
        self.__score_fnc = score_fnc

    @property
    def targets(self):
        return self.__validation_set.Y

    def evaluate(self, training_strategy: TrainingStrategy) -> float:
        model = training_strategy.train(self.__train_set, self.__validation_set)
        train_predictions = model.predict(self.__train_set.X)
        train_score = self.__score_fnc.compute_score(predictions=train_predictions, targets=self.__train_set.Y)
        if self.__validation_set is not None:
            predictions = model.predict(self.__validation_set.X)
            score = self.__score_fnc.compute_score(predictions=predictions, targets=self.__validation_set.Y)
        else:
            predictions = None
            score = 0

        return predictions, score
