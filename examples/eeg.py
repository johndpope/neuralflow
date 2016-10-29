import numpy
import numpy as np
import tensorflow as tf
from neuralflow.math_utils import norm
from neuralflow.models.Model import ExternalInputModel
from neuralflow import BatchProducer
from neuralflow import GaussianInitialization
from neuralflow import ValidationProducer

from neuralflow.neuralnets.FeedForwardNeuralNet import FeedForwardNeuralNet
from neuralflow.neuralnets.FeedForwardNeuralNet import StandardLayerProducer
from neuralflow.optimization.CustomOptimizer import CustomOptimizer
from neuralflow.optimization.Monitor import ScalarMonitor, RocMonitor
from neuralflow.optimization.StoppingCriterion import ThresholdCriterion
from neuralflow.neuralnets.ActivationFunction import SoftmaxActivationFunction, IdentityFunction, ReLUActivationFunction
from neuralflow.neuralnets.ActivationFunction import TanhActivationFunction
from neuralflow.neuralnets.LossFunction import CrossEntropy, SquaredError, MAE
from neuralflow.optimization.IterativeTraining import IterativeTraining
from neuralflow.optimization.SupervisedOptimizationProblem import SupervisedOptimizationProblem
from neuralflow.optimization.GradientDescent import GradientDescent
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import pickle


def add_dummy_dim(data):
    return np.reshape(data, newshape=(data.shape[0], 1))


class EegDataset(BatchProducer, ValidationProducer):
    def __init__(self, data, seed: int):
        x_train = data['X_train']
        x_validation = data['X_validation']
        x_test = data["X_test"]

        y_train = add_dummy_dim(data["Y_train"])
        y_validation = add_dummy_dim(data['Y_validation'])

        self.__pos = np.nonzero(y_train == 1)[0]
        self.__neg = np.nonzero(y_train == 0)[0]

        print("Y_train: ", y_train.shape)
        print("Y_val :", y_validation.shape)

        self.__y_train = y_train
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

        balanced = False
        if balanced:
            bs = int(batch_size / 2)
            pos_i = self.__rnd.randint(0, len(self.__pos), size=(bs,))
            neg_i = self.__rnd.randint(0, len(self.__neg), size=(bs,))
            indexes = np.concatenate((self.__pos[pos_i], self.__neg[neg_i]))
        else:
            indexes = self.__rnd.randint(0, self.__x_train.shape[0], size=(batch_size,))

        batch = {
            'output': self.__y_train[indexes],
            'input': self.__x_train[indexes]
        }

        return batch


# Data sets

data_file = "/home/giulio/neural_eeg.pkl"
data = pickle.load(open(data_file, "rb"))
dataset = EegDataset(data=data, seed=12)

# train
example = dataset.get_batch(20)
n_in, n_out = example["input"].shape[1], example["output"].shape[1]

seed = 13

hidden_layer_prod_1 = StandardLayerProducer(n_units=500, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                            activation_fnc=TanhActivationFunction())

hidden_layer_prod_2 = StandardLayerProducer(n_units=10, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                            activation_fnc=TanhActivationFunction())
output_layer_prod = StandardLayerProducer(n_units=n_out, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                          activation_fnc=SoftmaxActivationFunction(single_output=True))
# net1 = FeedForwardNeuralNet(input_model=ExternalInputModel(n_in=n_in), layer_producers=[hidden_layer_prod_1, hidden_layer_prod_2])


net = FeedForwardNeuralNet(input_model=ExternalInputModel(n_in=n_in),
                           layer_producers=[hidden_layer_prod_1, hidden_layer_prod_1, output_layer_prod])

print("n_in:{}, n_out:{}".format(n_in, n_out))

batch_producer = dataset
validation_producer = batch_producer
# objective function
loss_fnc = CrossEntropy(single_output=True)

problem = SupervisedOptimizationProblem(model=net, loss_fnc=loss_fnc, batch_producer=batch_producer, batch_size=30)
# optimizer
optimizer = GradientDescent(lr=0.01, problem=problem)

grad_monitor = ScalarMonitor(name="grad_norm", variable=norm(optimizer.gradient, norm_type="l2"))

# training
training = IterativeTraining(max_it=10 ** 6, optimizer=optimizer, problem=problem)

roc_monitor = RocMonitor(prediction=net.output, labels=problem.labels)

loss_monitor = ScalarMonitor(name="Loss", variable=problem.objective_fnc_value)

stopping_criterion = ThresholdCriterion(monitor=loss_monitor, thr=0.2, direction='<')

training.add_monitors(monitors=[grad_monitor], freq=100, name="batch")

validation_batch = validation_producer.get_validation()
training.add_monitors(monitors=[loss_monitor, roc_monitor], freq=100, name="validation",
                      feed_dict={net.input: validation_batch["input"], problem.labels: validation_batch["output"]})

train_batch = dataset.get_train()
training.add_monitors(monitors=[loss_monitor, roc_monitor], freq=100, name="train",
                      feed_dict={net.input: train_batch["input"], problem.labels: train_batch["output"]})

training.set_stopping_criterion([stopping_criterion])

training.train()
