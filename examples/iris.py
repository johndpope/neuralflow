import numpy
import numpy as np
import tensorflow as tf
from SessionManager import SessionManager
from math_utils import norm
from models.Model import ExternalInputModel
from neuralflow import BatchProducer
from neuralflow import GaussianInitialization
from neuralflow import ValidationProducer

from neuralflow.neuralnets.FeedForwardNeuralNet import FeedForwardNeuralNet
from neuralflow.neuralnets.FeedForwardNeuralNet import StandardLayerProducer
from neuralflow.optimization.CustomOptimizer import CustomOptimizer
from neuralflow.optimization.Monitor import ScalarMonitor, AccuracyMonitor
from neuralflow.optimization.Criterion import ThresholdCriterion
from neuralflow.neuralnets.ActivationFunction import SoftmaxActivationFunction
from neuralflow.neuralnets.ActivationFunction import TanhActivationFunction
from neuralflow.neuralnets.LossFunction import CrossEntropy
from neuralflow.optimization.IterativeTraining import IterativeTraining
from neuralflow.optimization.SupervisedOptimizationProblem import SupervisedOptimizationProblem
from optimization.GradientDescent import GradientDescent
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


class IrisDataset(BatchProducer, ValidationProducer):
    def __init__(self, csv_path: str, seed: int):
        training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=csv_path + "iris_training.csv",
                                                               target_dtype=np.int, features_dtype=np.float)
        test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=csv_path + "iris_test.csv", target_dtype=np.int, features_dtype=np.float)

        x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
                                           training_set.target, test_set.target

        scaler = preprocessing.StandardScaler().fit(x_train)
        x_test = scaler.transform(x_test)
        x_train = scaler.transform(x_train)

        enc = OneHotEncoder(sparse=False)
        y_train = enc.fit_transform(numpy.reshape(y_train, (y_train.shape[0], 1)))
        y_test = enc.transform(numpy.reshape(y_test, (y_test.shape[0], 1)))

        n = x_train.shape[0]
        n_val = int(0.3 * n)

        self.__validation_batch = {
            'output': y_train[(n - n_val):],
            'input': x_train[(n - n_val):, :]
        }
        self.__x_train = x_train[0:(n - n_val), :]
        self.__y_train = y_train[0:(n - n_val)]

        self.__test_batch = {
            'output': y_test,
            'input': x_test
        }

        self.__rnd = np.random.RandomState(seed)

    def get_validation(self):
        return self.__validation_batch

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
csv_path = "./examples/"
dataset = IrisDataset(csv_path=csv_path, seed=12)

# train
example = dataset.get_batch(1)
n_in, n_out = example["input"].shape[1], example["output"].shape[1]
print("n_in:{}, n_out:{}".format(n_in, n_out))

seed = 13

hidden_layer_prod = StandardLayerProducer(n_units=50, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                          activation_fnc=TanhActivationFunction())
output_layer_prod = StandardLayerProducer(n_units=n_out, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                          activation_fnc=SoftmaxActivationFunction(single_output=False))

net = FeedForwardNeuralNet(input_model=ExternalInputModel(n_in=n_in), layer_producers=[hidden_layer_prod, hidden_layer_prod, output_layer_prod])

batch_producer = dataset
validation_producer = batch_producer
# objective function
loss_fnc = CrossEntropy(single_output=False)

problem = SupervisedOptimizationProblem(model=net, loss_fnc=loss_fnc, batch_producer=batch_producer, batch_size=20)
# optimizer
optimizer = GradientDescent(lr=0.01, problem=problem)
# training
training = IterativeTraining(max_it=10 ** 6, optimizer=optimizer, problem=problem)

# monitors
grad_monitor = ScalarMonitor(name="grad_norm", variable=norm(optimizer.gradient, norm_type="l2"))
accuracy_monitor = AccuracyMonitor(predictions=net.output, labels=problem.labels)
loss_monitor = ScalarMonitor(name="loss", variable=problem.objective_fnc_value)

stopping_criterion = ThresholdCriterion(monitor=loss_monitor, thr=0.2, direction='<')

validation_batch = validation_producer.get_validation()
training.add_monitors_and_criteria(monitors=[loss_monitor, accuracy_monitor], freq=100, name="validation",
                                   feed_dict={net.input: validation_batch["input"], problem.labels: validation_batch["output"]})
training.add_monitors_and_criteria(monitors=[grad_monitor], freq=100, name="train_batch")

training.set_stopping_criterion([stopping_criterion])

training.train()

sess = SessionManager.get_session()

out = sess.run(net.output, feed_dict={net.input:validation_batch["input"]})
print(out)
