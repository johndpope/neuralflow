from neuralflow import BatchProducer
from neuralflow import ValidationProducer
from scipy.io import loadmat
from sklearn import preprocessing
import numpy as np


class EnelDataset(BatchProducer, ValidationProducer):

    # @staticmethod
    # def slice_hour(hour, x, y):
    #
    #     n = y.shape[0]
    #     indexes = np.arange(hour, n, 24)
    #     return x[indexes], y[indexes]

    @staticmethod
    def slice_hours(hours, x, y):

        y_ = y[:, hours["output"]]
        if len(y_.shape) == 1:
            y_ = np.reshape(y_, newshape=(y_.shape[0], 1))
        return x[:, hours["input"]], y_

    def __init__(self, mat_file: str, seed: int, scale_y=True, name: str = "UnknownRegion", hours:dict=None):

        self.__name = name
        mat_obj = loadmat(mat_file)

        x_train = mat_obj['X_train']
        x_validation = mat_obj['X_validation']

        y_train = mat_obj['Y_train']
        y_validation = mat_obj['Y_validation']

        x_test = mat_obj['X_test']
        y_test = mat_obj['Y_test']

        print("Y_train: ", y_train.shape)
        print("Y_val :", y_validation.shape)
        print("Y_test :", y_test.shape)

        if hours is not None:
            x_train, y_train = EnelDataset.slice_hours(hours, x=x_train, y=y_train)
            x_validation, y_validation = EnelDataset.slice_hours(hours, x=x_validation, y=y_validation)
            x_test, y_test = EnelDataset.slice_hours(hours, x=x_test, y=y_test)
            self.__name += "_{}hh".format(hours["output"]+1)

        if scale_y:
            self.__y_scaler = preprocessing.StandardScaler().fit(y_train)
            self.__y_train = self.__y_scaler.transform(y_train)
            self.__y_validation = self.__y_scaler.transform(y_validation)
            self.__y_test = self.__y_scaler.transform(y_test)

        else:
            self.__y_train = y_train
            self.__y_test = y_test
            self.__y_validation = y_validation

        x_scaler = preprocessing.StandardScaler().fit(x_train)
        self.__x_test = x_scaler.transform(x_test)
        self.__x_train = x_scaler.transform(x_train)
        self.__x_validation = x_scaler.transform(x_validation)

        self.__validation_batch = {
            'output': self.__y_validation,
            'input': self.__x_validation
        }

        self.__test_batch = {
            'output': self.__y_test,
            'input': self.__x_test
        }

        self.__train_batch = {
            'output': self.__y_train,
            'input': self.__x_train
        }

        self.__rnd = np.random.RandomState(seed)

    def get_validation(self):
        return self.__validation_batch

    def get_train(self):
        return self.__train_batch

    def get_test(self):
        return self.__test_batch

    def get_batch(self, batch_size):
        indexes = self.__rnd.randint(0, self.__x_train.shape[0], size=(batch_size,))

        batch = {
            'output': self.__y_train[indexes],
            'input': self.__x_train[indexes]
        }

        return batch

    @property
    def y_scaler(self):
        return self.__y_scaler

    @property
    def name(self):
        return self.__name
