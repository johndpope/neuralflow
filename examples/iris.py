import numpy
import numpy as np
import tensorflow as tf
from models.Model import Model
from monitors.CheckPointer import CheckPointer
from monitors.Criteria import MaxNoImproveCriterion, ImprovedValueCriterion
from monitors.Quantity import ExternalFeed, PrimitiveQuantity, AccuracyMonitor, ScalarMonitor
from monitors.logging_utils import start_logger
from neuralflow.TensorInitilization import GaussianInitialization
from neuralflow.neuralnets.ActivationFunction import SoftmaxActivationFunction
from neuralflow.neuralnets.ActivationFunction import TanhActivationFunction
from neuralflow.neuralnets.FeedForwardNeuralNet import FeedForwardNeuralNet
from neuralflow.neuralnets.LossFunction import CrossEntropy
from neuralflow.optimization.IterativeTraining import IterativeTraining
from neuralflow.optimization.SupervisedOptimizationProblem import SupervisedOptimizationProblem
from neuralnets.Layers import StandardLayerProducer, RBFLayerProducer
from optimization.Algorithm import SimpleAlgorithm
from optimization.GradientDescent import GradientDescent
from optimization.Penalty import NormPenalty
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from utils.Dataset import ValidationProducer, BatchProducer


class IrisDataset(BatchProducer, ValidationProducer):
    def __init__(self, csv_path: str, seed: int):
        training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=csv_path + "iris_training.csv",
                                                                           target_dtype=np.int, features_dtype=np.float)
        test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=csv_path + "iris_test.csv",
                                                                       target_dtype=np.int, features_dtype=np.float)

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


def define_problem(dataset, output_dir, logger):
    example = dataset.get_batch(1)
    n_in, n_out = example["input"].shape[1], example["output"].shape[1]
    print("n_in:{}, n_out:{}".format(n_in, n_out))

    seed = 13

    hidden_layer_prod = StandardLayerProducer(n_units=50, initialization=GaussianInitialization(mean=0, std_dev=0.01),
                                              activation_fnc=TanhActivationFunction())
    rbf_layer_producer = RBFLayerProducer(n_units=10, initialization=GaussianInitialization(mean=0, std_dev=0.01))
    output_layer_prod = StandardLayerProducer(n_units=n_out, initialization=GaussianInitialization(mean=0, std_dev=0.01),
                                              activation_fnc=SoftmaxActivationFunction(single_output=False))

    net = FeedForwardNeuralNet(n_in=n_in)
    net.add_layer(hidden_layer_prod)
    net.add_layer(hidden_layer_prod)
    # net.add_layer(rbf_layer_producer)
    net.add_layer(output_layer_prod)

    external_input = Model.from_external_input(n_in=n_in)

    model = Model.from_fnc(model=external_input, fnc=net)

    # objective function
    loss_fnc = CrossEntropy(single_output=False)

    # penalty
    p = NormPenalty(quantities_tf=model.trainables)
    penalty = (1000, p)

    # problem
    problem = SupervisedOptimizationProblem(model=model, loss_fnc=loss_fnc, batch_producer=dataset, batch_size=20,
                                            penalty=penalty)
    # optimizer
    optimizing_step = GradientDescent(lr=0.01)

    algorithm = SimpleAlgorithm(problem=problem, optimization_step=optimizing_step)

    # monitors
    # validation feed
    validation_batch = dataset.get_validation()
    validation_feed = ExternalFeed(feed_dict={model.input: validation_batch["input"],
                                              problem.labels: validation_batch["output"]}, freq=100,
                                   output_dir=output_dir,
                                   name="validation")

    validation_y = PrimitiveQuantity(model.output, name="y_val", feed=validation_feed)
    acc_val_monitor = AccuracyMonitor(validation_batch["output"], logger=logger)
    validation_y.register(acc_val_monitor)

    # saving criteria
    value_improved_criterion = ImprovedValueCriterion(monitor=acc_val_monitor, direction=">")
    checkpointer = CheckPointer(output_dir=output_dir, logger=logger, save_criterion=value_improved_criterion,
                                algorithm=algorithm)

    # training
    training = IterativeTraining(max_it=10 ** 6, algorithm=algorithm, output_dir=output_dir,
                                 feed_dicts=[validation_feed], logger=logger, check_pointer=checkpointer)

    max_no_improve = MaxNoImproveCriterion(max_no_improve=50, direction="<", monitor=acc_val_monitor, logger=logger)
    training.set_stop_criterion(max_no_improve)

    # iterative feed
    penalty_it = PrimitiveQuantity(p.value_tf, name="penalty", feed=training.iterative_feed)
    # penalty_it = PrimitiveQuantity(tf.norm(algorithm.gradient, 2), name="grad", feed=training.iterative_feed)

    penalty_monitor = ScalarMonitor(name="penalty", logger=logger, format=":.3f")
    penalty_it.register(penalty_monitor)

    obj_fnc = PrimitiveQuantity(problem.objective_fnc_value, name="obj_value",
                                      feed=training.iterative_feed)
    obj_monitor = ScalarMonitor(name="ObjFnc", logger=logger, format=":.2f")
    obj_fnc.register(obj_monitor)

    return training, model


if __name__ == "__main__":
    # Data sets
    csv_path = "./examples/"
    dataset = IrisDataset(csv_path=csv_path, seed=12)

    output_dir = "/home/galvan/tensorBoards/iris/"
    logger = start_logger(log_dir=output_dir, log_file="iris_train.log")

    training, model = define_problem(dataset, output_dir, logger)
    sess = tf.Session()
    training.train(sess=sess)

    new_saver = tf.train.import_meta_graph(output_dir + 'best_checkpoint.meta')
    new_saver.restore(sess, output_dir + 'best_checkpoint')

    net_out = tf.get_collection("model.out")[0]  # XXX
    net_in = tf.get_collection("model.in")[0]
    out = sess.run(model.output, feed_dict={model.input: dataset.get_test()["input"]})

    score = accuracy_score(y_true=numpy.argmax(dataset.get_test()["output"], axis=1), y_pred=numpy.argmax(out, axis=1))
    sess.close()
    print("Accuracy score: {:.2f}".format(score))
