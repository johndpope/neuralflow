import numpy
import numpy as np
import tensorflow as tf
from neuralflow.math_utils import norm
from neuralflow.models.Model import ExternalInputModel
from neuralflow import BatchProducer
from neuralflow import GaussianInitialization
from neuralflow import ValidationProducer

from neuralflow.neuralnets.FeedForwardNeuralNet import FeedForwardNeuralNet
from neuralnets.Layers import StandardLayerProducer
from neuralflow.optimization.CustomOptimizer import CustomOptimizer
from neuralflow.optimization.Monitor import ScalarMonitor
from neuralflow.optimization.Criterion import ThresholdCriterion
from neuralflow.neuralnets.ActivationFunction import SoftmaxActivationFunction, IdentityFunction, ReLUActivationFunction
from neuralflow.neuralnets.ActivationFunction import TanhActivationFunction
from neuralflow.neuralnets.LossFunction import CrossEntropy, SquaredError, MAE
from neuralflow.optimization.IterativeTraining import IterativeTraining
from neuralflow.optimization.SupervisedOptimizationProblem import SupervisedOptimizationProblem
from neuralflow.optimization.GradientDescent import GradientDescent
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


class EnelDataset(BatchProducer, ValidationProducer):
    def __init__(self, mat_file: str, seed: int, scale_y=True):
        mat_obj = loadmat(mat_file)

        x_train = mat_obj['X_train']
        x_validation = mat_obj['X_validation']

        y_train = mat_obj['Y_train']
        y_validation = mat_obj['Y_validation']

        print("Y_train: ", y_train.shape)
        print("Y_val :", y_validation.shape)

        mat_obj = loadmat("/home/giulio/SICI_by_hour_preprocessed.mat")
        x_test = mat_obj['X_test']
        y_test = mat_obj['Y_test']

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


scale_y = True
# Data sets
dataset = EnelDataset(mat_file="/home/giulio/SICI_train_and_val_cleaned_new.mat", seed=12, scale_y=scale_y)
#dataset = EnelDataset(mat_file="/home/giulio/SICI_by_hour_preprocessed.mat", seed=12, scale_y=scale_y)

# train
example = dataset.get_batch(1)
n_in, n_out = example["input"].shape[1], example["output"].shape[1]

seed = 13

hidden_layer_prod_1 = StandardLayerProducer(n_units=30, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                            activation_fnc=TanhActivationFunction())

hidden_layer_prod_2 = StandardLayerProducer(n_units=10, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                            activation_fnc=TanhActivationFunction())
output_layer_prod = StandardLayerProducer(n_units=n_out, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                          activation_fnc=IdentityFunction())
# net1 = FeedForwardNeuralNet(input_model=ExternalInputModel(n_in=n_in), layer_producers=[hidden_layer_prod_1, hidden_layer_prod_2])


net = FeedForwardNeuralNet(input_model=ExternalInputModel(n_in=n_in),
                           layer_producers=[hidden_layer_prod_1, output_layer_prod])

print("n_in:{}, n_out:{}".format(n_in, n_out))

batch_producer = dataset
validation_producer = batch_producer
# objective function
loss_fnc = MAE()

problem = SupervisedOptimizationProblem(model=net, loss_fnc=loss_fnc, batch_producer=batch_producer, batch_size=20)
# optimizer
optimizer = GradientDescent(lr=0.01, problem=problem)

grad_monitor = ScalarMonitor(name="grad_norm", variable=norm(optimizer.gradient, norm_type="l2"))

# training
training = IterativeTraining(max_it=10 ** 6, optimizer=optimizer, problem=problem)

# roc_monitor = RocMonitor(prediction=net.output, labels=training.labels)
# print(dataset.get_train()["output"].shape)
# mean = tf.constant(np.reshape(dataset.y_scaler.mean_, [1, 24]).astype(dtype='float32'))
# print(mean.get_shape())
# print(problem.objective_fnc_value.get_shape())
#
# scaled_variable = tf.sub(problem.objective_fnc_value, mean)#/dataset.y_scaler.scale_

if scale_y:
    mean = dataset.y_scaler.mean_.item()
    scale = dataset.y_scaler.scale_.item()
    print("Scale: {:.2f}".format(scale))
    print("Mean: {:.2f}".format(mean))


def scale_placeholder(x):
    if scale_y:
        return (x * scale) + mean
    else:
        return x

loss_monitor = ScalarMonitor(name="Loss", variable=problem.objective_fnc_value)

mae_loss = MAE(scale_placeholder)
mae = mae_loss.value(net.output, problem.labels)
mae_monitor = ScalarMonitor(name="MAE", variable=mae)

stopping_criterion = ThresholdCriterion(monitor=loss_monitor, thr=0.2, direction='<')

training.add_monitors_and_criteria(monitors=[grad_monitor], freq=100, name="batch")

validation_batch = validation_producer.get_validation()
training.add_monitors_and_criteria(monitors=[loss_monitor], freq=100, name="validation",
                                   feed_dict={net.input: validation_batch["input"], problem.labels: validation_batch["output"]})

train_batch = dataset.get_train()
training.add_monitors_and_criteria(monitors=[mae_monitor], freq=100, name="train",
                                   feed_dict={net.input: train_batch["input"], problem.labels: train_batch["output"]})

test_batch = dataset.get_test()
training.add_monitors_and_criteria(monitors=[mae_monitor], freq=100, name="test",
                                   feed_dict={net.input: test_batch["input"], problem.labels: test_batch["output"]})
training.set_stopping_criterion([stopping_criterion])

training.train()
