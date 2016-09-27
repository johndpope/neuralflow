import numpy
import tensorflow as tf
import numpy as np
from neuralflow.CustomOptimizer import CustomOptimizer
from neuralflow.StoppingCriterion import ThresholdCriterion
from neuralflow.FeedForwardNeuralNet import FeedForwardNeuralNet
from neuralflow.FeedForwardNeuralNet import StandardLayerProducer
from neuralflow.Monitor import RocMonitor
from neuralflow.Monitor import ScalarMonitor
from neuralflow import BatchProducer
from neuralflow import CrossEntropy
from neuralflow import GaussianInitialization
from neuralflow import IterativeTraining
from neuralflow import SoftmaxActivationFunction
from neuralflow import TanhActivationFunction
from neuralflow import ValidationProducer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


class IrisDataset(BatchProducer, ValidationProducer):
    def __init__(self, csv_path: str, seed: int):
        training_set = tf.contrib.learn.datasets.base.load_csv(filename=csv_path + "iris_training.csv",
                                                               target_dtype=np.int)
        test_set = tf.contrib.learn.datasets.base.load_csv(filename=csv_path + "iris_test.csv", target_dtype=np.int)

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
csv_path = "examples/"
dataset = IrisDataset(csv_path=csv_path, seed=12)

# train
example = dataset.get_batch(1)
n_in, n_out = example["input"].shape[1], example["output"].shape[1]

seed = 13

hidden_layer_prod = StandardLayerProducer(n_units=50, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                          activation_fnc=TanhActivationFunction())
output_layer_prod = StandardLayerProducer(n_units=n_out, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                          activation_fnc=SoftmaxActivationFunction(single_output=True))
net = FeedForwardNeuralNet(n_in=n_in, layer_producers=[hidden_layer_prod, hidden_layer_prod, output_layer_prod])

batch_producer = dataset
validation_producer = batch_producer
# objective function
obj_fnc = CrossEntropy(single_output=False)

# optimizer
optimizer = CustomOptimizer(0.001)

# training
training = IterativeTraining(batch_size=100, obj_fnc=obj_fnc, model=net, batch_producer=batch_producer,
                             validation_producer=validation_producer, max_it=10**6,
                             optimizer=optimizer)

#roc_monitor = RocMonitor(prediction=net.output, labels=training.labels)
loss_monitor = ScalarMonitor(name="loss", variable=training.loss)

stopping_criterion = ThresholdCriterion(monitor=loss_monitor, thr=0.2, direction='<')


validation_batch = validation_producer.get_validation()
training.set_monitors(monitors=[loss_monitor], freq=100, name="validation",
                      feed_dict={net.input: validation_batch["input"], training.labels: validation_batch["output"]})

training.set_stopping_criterion([stopping_criterion])

training.train()
