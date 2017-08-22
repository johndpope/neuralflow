import numpy as np
import tensorflow as tf
from exploration.ArtificialDataset import ArtificialDataset
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
from neuralnets.Layers import StandardLayerProducer
from optimization.Algorithm import SimpleAlgorithm
from optimization.GradientDescent import GradientDescent
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils.Dataset import ValidationProducer, BatchProducer
from utils.dataset_utils import add_dummy_dim


class ExplDataset(BatchProducer, ValidationProducer):
    def __init__(self, seed: int, n_feats: int, n_samples, n_informative: int):
        data = ArtificialDataset(seed=seed, n_feats=n_feats, n_samples=n_samples, n_informative=n_informative)

        x_train, x_test, y_train, self.__y_test = train_test_split(data.X, data.Y, test_size=0.30, random_state=seed)

        scaler = preprocessing.StandardScaler().fit(x_train)
        self.__x_test = scaler.transform(x_test)
        x_train = scaler.transform(x_train)

        self.__x_train, self.__x_val, self.__y_train, self.__y_val = train_test_split(x_train, y_train, test_size=0.30,
                                                                                      random_state=seed)

        print("n_feats: {}, n_train: {}, n_val:{}, n_test:{}".format(n_feats, x_train.shape[0], self.__x_val.shape[0],
                                                                     self.__x_test.shape[0]))

        self.__y_test = add_dummy_dim(self.__y_test)
        self.__y_val = add_dummy_dim(self.__y_val)
        self.__y_train = add_dummy_dim(self.__y_train)

        self.__rnd = np.random.RandomState(seed)

    def get_validation(self):
        return {
            'output': self.__y_val,
            'input': self.__x_val
        }

    def get_train(self):
        return {
            'output': self.__y_train,
            'input': self.__x_train
        }

    def get_test(self):
        return {
            'output': self.__y_test,
            'input': self.__x_test
        }

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

    hidden_layer_prod = StandardLayerProducer(n_units=200, initialization=GaussianInitialization(mean=0, std_dev=0.01),
                                              activation_fnc=TanhActivationFunction())
    output_layer_prod = StandardLayerProducer(n_units=n_out,
                                              initialization=GaussianInitialization(mean=0, std_dev=0.01),
                                              activation_fnc=SoftmaxActivationFunction(single_output=True))

    net = FeedForwardNeuralNet(n_in=n_in)
    # net.add_layer(hidden_layer_prod)
    net.add_layer(hidden_layer_prod)
    net.add_layer(output_layer_prod)

    external_input = Model.from_external_input(n_in=n_in)

    model = Model.from_fnc(model=external_input, fnc=net)

    # objective function
    loss_fnc = CrossEntropy(single_output=True)

    problem = SupervisedOptimizationProblem(model=model, loss_fnc=loss_fnc, batch_producer=dataset, batch_size=20)
    # optimizer
    optimizing_step = GradientDescent(lr=0.01)

    algorithm = SimpleAlgorithm(problem=problem, optimization_step=optimizing_step)

    # monitors
    # grad_monitor = ScalarMonitor(name="grad_norm", variable=norm(algorithm.gradient, norm_type="l2"))

    # validation feed
    validation_batch = dataset.get_validation()
    validation_feed = ExternalFeed(feed_dict={model.input: validation_batch["input"],
                                              problem.labels: validation_batch["output"]}, freq=100,
                                   output_dir=output_dir,
                                   name="validation")

    validation_y = PrimitiveQuantity(model.output, name="y_val", feed=validation_feed)
    acc_val_monitor = AccuracyMonitor(validation_batch["output"], logger=logger)
    validation_y.register(acc_val_monitor)

    # training feed
    train_batch = dataset.get_train()
    train_feed = ExternalFeed(feed_dict={model.input: train_batch["input"],
                                         problem.labels: train_batch["output"]}, freq=100, output_dir=output_dir,
                              name="train")

    train_y = PrimitiveQuantity(model.output, name="y_tr", feed=train_feed)
    acc_tr_monitor = AccuracyMonitor(train_batch["output"], logger=logger)
    train_y.register(acc_tr_monitor)

    loss_tr = PrimitiveQuantity(problem.objective_fnc_value, name="loss_tr", feed=train_feed)
    loss_tr_monitor = ScalarMonitor(name="Loss", logger=logger)
    loss_tr.register(loss_tr_monitor)

    # saving criteria
    value_improved_criterion = ImprovedValueCriterion(monitor=acc_val_monitor, direction=">")
    checkpointer = CheckPointer(output_dir=output_dir, logger=logger, save_criterion=value_improved_criterion,
                                algorithm=algorithm)

    # training
    training = IterativeTraining(max_it=10 ** 6, algorithm=algorithm, output_dir=output_dir,
                                 feed_dicts=[validation_feed], logger=logger, check_pointer=checkpointer)

    # iterative batch feed
    gradient_norm = PrimitiveQuantity(tf.norm(algorithm.gradient, ord=2), name="grad_norm",
                                      feed=training.iterative_feed)
    loss_tr_monitor = ScalarMonitor(name="GradientNorm", logger=logger, format=":.2e")
    gradient_norm.register(loss_tr_monitor)

    max_no_improve = MaxNoImproveCriterion(max_no_improve=200, direction="<", monitor=acc_val_monitor, logger=logger)
    training.set_stop_criterion(max_no_improve)

    return training, model


if __name__ == "__main__":
    # Data sets
    dataset = ExplDataset(seed=12, n_feats=10, n_samples=100000, n_informative=3)

    output_dir = "/home/galvan/tensorBoards/expl/"
    logger = start_logger(log_dir=output_dir, log_file="train.log")

    training, model = define_problem(dataset, output_dir, logger)
    sess = tf.Session()
    training.train(sess=sess)

    new_saver = tf.train.import_meta_graph(output_dir + 'best_checkpoint.meta')
    new_saver.restore(sess, output_dir + 'best_checkpoint')

    net_out = tf.get_collection("model.out")[0]  # XXX
    net_in = tf.get_collection("model.in")[0]
    out = sess.run(model.output, feed_dict={model.input: dataset.get_test()["input"]})

    score = accuracy_score(y_true=dataset.get_test()["output"], y_pred=np.round(out))
    sess.close()
    print("Accuracy score: {:.2f}".format(score))
