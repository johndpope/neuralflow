from enel.svm.Dataset import Dataset
from enel.svm.Model import TrainingStrategy, Model
from sklearn.svm import SVR as _SVR


class SVRTrainingStrategy(TrainingStrategy):
    def __init__(self, gamma: float, C: float, epsilon: float):
        self.__gamma = gamma
        self.__C = C
        self.__epsilon = epsilon

    def train(self, training_set: Dataset, validation_set: Dataset = None):
        svr = _SVR(kernel='rbf', gamma=self.__gamma, C=self.__C, epsilon=self.__epsilon)
        svr.fit(training_set.X, training_set.Y)
        return SVR(svr)

    @property
    def gamma(self):
        return self.__gamma

    @property
    def C(self):
        return self.__C

    @property
    def epsilon(self):
        return self.__epsilon

    def __str__(self):
        return "SVR->gamma:{:.2e}, C:{:.2e}, epsilon: {:.2e}".format(self.__gamma, self.__C, self.__epsilon)


class SVR(Model):
    def __init__(self, svc):
        self.__svc = svc

    def predict(self, data):
        return self.__svc.predict(data)
