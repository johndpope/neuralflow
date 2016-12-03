from Kaggle.MachineLearning.Model import Model, TrainingStrategy
import os
import numpy as np
from Kaggle.Utils.data_manipulation import Dataset
from neuralflow.models.Model import ExternalInputModel
from neuralflow import BatchProducer, ValidationProducer
from neuralflow import GaussianInitialization
from neuralflow import HingeLoss
from neuralflow import SoftmaxActivationFunction
from neuralflow import TanhActivationFunction
from neuralflow.neuralnets.FeedForwardNeuralNet import StandardLayerProducer, FeedForwardNeuralNet
from neuralflow.optimization.Criterion import MaxNoImproveCriterion, ImprovedValueCriterion
from neuralflow.optimization.GradientDescent import GradientDescent
from neuralflow.optimization.IterativeTraining import IterativeTraining
from neuralflow.optimization.Monitor import ScalarMonitor, RocMonitor
from neuralflow.optimization.SupervisedOptimizationProblem import SupervisedOptimizationProblem
from neuralflow.math_utils import norm
import tensorflow as tf


def add_dummy_dim(data):
    return np.reshape(data, newshape=(data.shape[0], 1))


class KaggleEegDataset(BatchProducer, ValidationProducer):
    def __init__(self, data, target, validation_set: Dataset, seed: int):
        self.__x_train = data
        self.__x_validation = validation_set.X

        self.__y_train = add_dummy_dim(target)
        self.__y_validation = add_dummy_dim(validation_set.Y)

        self.__pos = np.nonzero(self.__y_train == 1)[0]
        self.__neg = np.nonzero(self.__y_train == 0)[0]

        print("Y_train: ", self.__y_train.shape)
        print("Y_val :", self.__y_validation.shape)

        self.__validation_batch = {
            'output': self.__y_validation,
            'input': self.__x_validation
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

    @property
    def class_weights(self):
        n_pos = len(np.nonzero(self.__train_batch["output"])[0])
        n_neg = len(np.nonzero(self.__train_batch["output"] < 1)[0])

        weights = np.array((1. / n_neg, 1. / n_pos))
        weights = weights / sum(weights)
        return weights

    def get_batch(self, batch_size):

        balanced = True
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


class FNNTrainingStrategy(TrainingStrategy):
    def __init__(self, output_dir, id: int, seed: int, h:int, n:int):
        self.__output_dir = output_dir
        self.__id = id
        self.__count = -1
        self.__seed = seed
        self.__h = h
        self.__n = n

    def train(self, data, target, unlabeled, validation_set):

        tf.reset_default_graph()

        self.__count += 1
        save_dir = self.__output_dir + '/{}_{}/'.format(self.__id, self.__count)
        os.makedirs(save_dir, exist_ok=True)

        dataset = KaggleEegDataset(data, target, validation_set, self.__seed)
        # train
        example = dataset.get_batch(20)
        # print(example)
        n_in, n_out = example["input"].shape[1], example["output"].shape[1]

        seed = self.__seed

        hidden_layer_prod_1 = StandardLayerProducer(n_units=self.__h,
                                                    initialization=GaussianInitialization(mean=0, std_dev=0.1,
                                                                                          seed=seed),
                                                    activation_fnc=TanhActivationFunction())

        hidden_layer_prod_2 = StandardLayerProducer(n_units=2000,
                                                    initialization=GaussianInitialization(mean=0, std_dev=0.1,
                                                                                          seed=seed),
                                                    activation_fnc=TanhActivationFunction())

        hidden_layer_prod_3 = StandardLayerProducer(n_units=500,
                                                    initialization=GaussianInitialization(mean=0, std_dev=0.1,
                                                                                          seed=seed),
                                                    activation_fnc=TanhActivationFunction())

        hidden_layer_prod_4 = StandardLayerProducer(n_units=100,
                                                    initialization=GaussianInitialization(mean=0, std_dev=0.1,
                                                                                          seed=seed),
                                                    activation_fnc=TanhActivationFunction())
        output_layer_prod = StandardLayerProducer(n_units=n_out,
                                                  initialization=GaussianInitialization(mean=0, std_dev=0.1, seed=seed),
                                                  activation_fnc=SoftmaxActivationFunction(single_output=True))

        net = FeedForwardNeuralNet(input_model=ExternalInputModel(n_in=n_in))
        for _ in range(self.__n):
            net.add_layer(hidden_layer_prod_1)
            net.add_layer(hidden_layer_prod_1)
        # net.add_layer(hidden_layer_prod_2)
        # net.add_layer(hidden_layer_prod_2)
        # net.add_layer(hidden_layer_prod_3)
        # net.add_layer(hidden_layer_prod_3)
        # net.add_layer(hidden_layer_prod_3)
        # net.add_layer(hidden_layer_prod_4)
        # net.add_layer(hidden_layer_prod_4)
        # net.add_layer(hidden_layer_prod_4)
        net.add_layer(output_layer_prod)

        print("n_in:{}, n_out:{}".format(n_in, n_out))

        batch_producer = dataset
        validation_producer = batch_producer
        # objective function
        # loss_fnc = CrossEntropy(single_output=True)  # , class_weights=dataset.class_weights)
        loss_fnc = HingeLoss()

        problem = SupervisedOptimizationProblem(model=net, loss_fnc=loss_fnc, batch_producer=batch_producer,
                                                batch_size=20)
        # optimizer
        optimizer = GradientDescent(lr=0.05, problem=problem)

        grad_monitor = ScalarMonitor(name="grad_norm", variable=norm(optimizer.gradient, norm_type="l2"))

        # training
        training = IterativeTraining(max_it=10 * 4, optimizer=optimizer, problem=problem, output_dir=save_dir)

        # monitors
        tr_roc_monitor = RocMonitor(predictions=net.output, labels=problem.labels)
        tr_loss_monitor = ScalarMonitor(name="Loss", variable=problem.objective_fnc_value)

        val_roc_monitor = RocMonitor(predictions=net.output, labels=problem.labels)
        val_loss_monitor = ScalarMonitor(name="Loss", variable=problem.objective_fnc_value)

        # stopping_criteria
        # thr_criterion = ThresholdCriterion(monitor=loss_monitor, thr=0.2, direction='<')
        max_no_improve = MaxNoImproveCriterion(monitor=val_roc_monitor, max_no_improve=100, direction=">")

        # low_grad_stopping_criterion = ThresholdCriterion(monitor=grad_monitor, thr=0.01, direction="<")

        # saving criteria
        value_improved_criterion = ImprovedValueCriterion(monitor=val_roc_monitor, direction=">")

        training.add_monitors_and_criteria(monitors=[grad_monitor], freq=100, name="batch")

        # add monitors and criteria on validation-set
        validation_batch = validation_producer.get_validation()
        training.add_monitors_and_criteria(freq=100,
                                           name="validation",
                                           feed_dict={net.input: validation_batch["input"],
                                                      problem.labels: validation_batch["output"]},
                                           monitors=[val_loss_monitor, val_roc_monitor],
                                           saving_criteria=[value_improved_criterion],
                                           stopping_criteria=[max_no_improve])

        sess = tf.Session()

        training.train(sess)

        return KaggleModel(save_dir)


class KaggleModel(Model):
    def __init__(self, output_dir):
        self.__output_dir = output_dir

    def predict(self, X: np.array) -> np.array:
        tf.reset_default_graph()
        sess = tf.Session()

        new_saver = tf.train.import_meta_graph(self.__output_dir + 'best_checkpoint.meta')
        new_saver.restore(sess, self.__output_dir + 'best_checkpoint')

        net_out = tf.get_collection("net.out")[0]  # XXX
        net_in = tf.get_collection("net.in")[0]

        predictions = sess.run(net_out, feed_dict={net_in: X})
        sess.close()
        return predictions
