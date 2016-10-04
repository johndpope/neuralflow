import numpy
import numpy as np
import tensorflow as tf
from neuralflow import BatchProducer
from neuralflow import GaussianInitialization
from neuralflow import ValidationProducer

from neuralflow.neuralnets.FeedForwardNeuralNet import FeedForwardNeuralNet
from neuralflow.neuralnets.FeedForwardNeuralNet import StandardLayerProducer
from neuralflow.optimization.CustomOptimizer import CustomOptimizer
from neuralflow.optimization.Monitor import ScalarMonitor
from neuralflow.optimization.StoppingCriterion import ThresholdCriterion
from neuralflow.neuralnets.ActivationFunction import SoftmaxActivationFunction, IdentityFunction
from neuralflow.neuralnets.ActivationFunction import TanhActivationFunction
from neuralflow.neuralnets.LossFunction import CrossEntropy, SquaredError, MAE
from neuralflow.optimization.IterativeTraining import IterativeTraining
from neuralflow.optimization.SupervisedOptimizationProblem import SupervisedOptimizationProblem
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


class EnelDataset(BatchProducer, ValidationProducer):
    def __init__(self, mat_file: str, seed: int):
        mat_obj = loadmat(mat_file)

        x_train = mat_obj['X_train']
        x_validation = mat_obj['X_validation']
        x_test = mat_obj['X_test']

        self.__y_train = mat_obj['Y_train']
        y_validation = mat_obj['Y_validation']
        y_test = mat_obj['Y_test']

        scaler = preprocessing.StandardScaler().fit(x_train)
        self.__x_test = scaler.transform(x_test)
        self.__x_train = scaler.transform(x_train)
        self.__x_validation = scaler.transform(x_validation)

        self.__validation_batch = {  # TODO rimetti validation
            'output': y_validation,
            'input': x_validation
        }

        self.__test_batch = {
            'output': y_test,
            'input': x_test
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


# Data sets
dataset = EnelDataset(mat_file="/home/giulio/SUD.mat", seed=12)

# train
example = dataset.get_batch(1)
n_in, n_out = example["input"].shape[1], example["output"].shape[1]

seed = 13

hidden_layer_prod_1 = StandardLayerProducer(n_units=1000, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                            activation_fnc=TanhActivationFunction())

hidden_layer_prod_2 = StandardLayerProducer(n_units=300, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                            activation_fnc=TanhActivationFunction())
output_layer_prod = StandardLayerProducer(n_units=n_out, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                          activation_fnc=IdentityFunction())
net = FeedForwardNeuralNet(n_in=n_in, layer_producers=[hidden_layer_prod_1, hidden_layer_prod_1, output_layer_prod])

print("n_in:{}, n_out:{}".format(n_in, n_out))

batch_producer = dataset
validation_producer = batch_producer
# objective function
loss_fnc = MAE()

problem = SupervisedOptimizationProblem(model=net, loss_fnc=loss_fnc, batch_producer=batch_producer, batch_size=20)
# optimizer
optimizer = CustomOptimizer(0.2)

# training
training = IterativeTraining(problem=problem, max_it=10 ** 6, optimizer=optimizer)

# roc_monitor = RocMonitor(prediction=net.output, labels=training.labels)
loss_monitor = ScalarMonitor(name="mae", variable=problem.objective_fnc_value)

mae_loss = MAE()
mae = mae_loss.value(net.output, problem.labels)
mae_monitor = ScalarMonitor(name="mae", variable=mae)

stopping_criterion = ThresholdCriterion(monitor=loss_monitor, thr=0.2, direction='<')

validation_batch = validation_producer.get_validation()
training.add_monitors(monitors=[loss_monitor], freq=100, name="validation",
                      feed_dict={net.input: validation_batch["input"], problem.labels: validation_batch["output"]})

train_batch = dataset.get_train()
training.add_monitors(monitors=[mae_monitor], freq=100, name="train",
                      feed_dict={net.input: train_batch["input"], problem.labels: train_batch["output"]})

test_batch = dataset.get_test()
training.add_monitors(monitors=[mae_monitor], freq=100, name="test",
                      feed_dict={net.input: test_batch["input"], problem.labels: test_batch["output"]})
training.set_stopping_criterion([stopping_criterion])

training.train()
