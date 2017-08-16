import numpy as np
import tensorflow as tf
from feature_selection.BiLevelAlgorithm import BiLevelAlgorithm
from models.Model import Model
from neuralflow.TensorInitilization import GaussianInitialization
from neuralflow.neuralnets.ActivationFunction import SoftmaxActivationFunction
from neuralflow.neuralnets.ActivationFunction import TanhActivationFunction
from neuralflow.neuralnets.FeedForwardNeuralNet import FeedForwardNeuralNet
from neuralflow.neuralnets.LossFunction import HingeLoss
from neuralflow.optimization.Criterion import MaxNoImproveCriterion, ImprovedValueCriterion
from neuralflow.optimization.IterativeTraining import IterativeTraining
from neuralflow.optimization.Monitor import ScalarMonitor, AccuracyMonitor
from neuralflow.optimization.SupervisedOptimizationProblem import SupervisedOptimizationProblem
from neuralnets.Layers import StandardLayerProducer
from optimization.Algorithm import SimpleAlgorithm
from optimization.GradientDescent import GradientDescent
from selfea.datasets.ArtificialDataset import ArtificialDataset
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from utils.BatchSequencer import BatchSequencer
from utils.Dataset import ValidationProducer, BatchProducer


class BiLevelDataset(BatchProducer, ValidationProducer):

    def __add_dummy_dim(self, v):
        assert len(v.shape)==1
        return np.reshape(v, newshape=(v.shape[0], 1))

    def __init__(self, dataset, seed: int):
        scaler = preprocessing.StandardScaler().fit(dataset.X)
        x_train = scaler.transform(dataset.X)
        y_train = dataset.Y

        n = x_train.shape[0]
        n_val = int(0.3 * n)

        self.__y_val = self.__add_dummy_dim(y_train[(n - n_val):])
        self.__x_val = x_train[(n - n_val):, :]
        self.__x_train = x_train[0:(n - n_val), :]
        self.__y_train = self.__add_dummy_dim(y_train[0:(n - n_val)])

        self.__rnd = np.random.RandomState(seed)

    @property
    def train(self):
        return self.__x_train, self.__y_train

    @property
    def validation(self):
        return self.__x_val, self.__y_val


def define_problem(dataset: BiLevelDataset, output_dir):
    seed = 13

    train_sequencer = BatchSequencer(X=dataset.train[0], Y=dataset.train[1], seed=seed)
    validation_sequencer = BatchSequencer(X=dataset.validation[0], Y=dataset.validation[1], seed=seed)

    example = train_sequencer.get_batch(1)
    n_in, n_out = example["input"].shape[1], example["output"].shape[1]
    print("n_in:{}, n_out:{}".format(n_in, n_out))

    external_input = Model.from_external_input(n_in=n_in)
    # feat_sel_fnc = FeedForwardNeuralNet(n_in=n_in).add_layer(
    #     ElementwiseMulLayerProducer(initialization=GaussianInitialization()))
    feat_sel_fnc = FeedForwardNeuralNet(n_in=n_in).add_layer(StandardLayerProducer(n_units=n_in, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                              activation_fnc=TanhActivationFunction()))
    part_model = Model.from_fnc(external_input, feat_sel_fnc)

    hidden_layer_prod = StandardLayerProducer(n_units=100, initialization=GaussianInitialization(mean=0, std_dev=0.01),
                                              activation_fnc=TanhActivationFunction())
    output_layer_prod = StandardLayerProducer(n_units=n_out, initialization=GaussianInitialization(mean=0, std_dev=0.01),
                                              activation_fnc=SoftmaxActivationFunction(single_output=True))

    net = FeedForwardNeuralNet(n_in=n_in)
    net.add_layer(hidden_layer_prod)
    net.add_layer(hidden_layer_prod)
    net.add_layer(output_layer_prod)

    model = Model.from_fnc(model=external_input, fnc=net)

    # objective function
    # loss_fnc = CrossEntropy(single_output=True)
    loss_fnc = HingeLoss()

    leader_problem = SupervisedOptimizationProblem(model=model, loss_fnc=loss_fnc, batch_producer=validation_sequencer,
                                                   batch_size=20, trainables=feat_sel_fnc.trainables, penalty=False)
    follower_problem = SupervisedOptimizationProblem(model=model, loss_fnc=loss_fnc,
                                                     batch_producer=train_sequencer, batch_size=20, trainables=net.trainables)
    # optimizer
    optimizing_step = GradientDescent(lr=0.05)

    algorithm = BiLevelAlgorithm(follower=follower_problem, leader=leader_problem, optimization_step=optimizing_step)

    algorithm = SimpleAlgorithm(problem=follower_problem, optimization_step=optimizing_step)
    # training
    training = IterativeTraining(max_it=10 ** 6, algorithm=algorithm, output_dir=output_dir)

    # monitors
    # grad_monitor = ScalarMonitor(name="grad_norm", variable=norm(algorithm.gradient, norm_type="l2"))
    accuracy_monitor = AccuracyMonitor(predictions=model.output, labels=leader_problem.labels)
    loss_monitor = ScalarMonitor(name="loss", variable=leader_problem.objective_fnc_value)

    max_no_improve = MaxNoImproveCriterion(monitor=loss_monitor, max_no_improve=50, direction="<")

    # saving criteria
    value_improved_criterion = ImprovedValueCriterion(monitor=loss_monitor, direction="<")

    # stopping_criterion = ThresholdCriterion(monitor=loss_monitor, thr=0.2, direction='<')

    validation_X, validation_Y = dataset.validation
    train_X, train_Y = dataset.train
    training.add_monitors_and_criteria(monitors=[loss_monitor], freq=100, name="train", feed_dict={model.input: train_X,
                                                  leader_problem.labels: train_Y})

    training.add_monitors_and_criteria(monitors=[loss_monitor, accuracy_monitor], freq=100, name="validation",
                                       feed_dict={model.input: validation_X,
                                                  leader_problem.labels: validation_Y},
                                       stopping_criteria=[max_no_improve], saving_criteria=[value_improved_criterion])

    return training, model, feat_sel_fnc.trainables


if __name__ == "__main__":
    # Data sets
    seed = 13
    data = ArtificialDataset(n_samples=20000, n_feats=50, n_informative=50, seed=seed)
    dataset = BiLevelDataset(data, seed)

    output_dir = "/home/galvan/tensorBoards/bilevel/"

    training, model, beta = define_problem(dataset, output_dir)
    sess = tf.Session()
    training.train(sess=sess)
    # sess.close()

    new_saver = tf.train.import_meta_graph(output_dir + 'best_checkpoint.meta')
    new_saver.restore(sess, output_dir + 'best_checkpoint')

    net_out = tf.get_collection("model.out")[0]  # XXX
    net_in = tf.get_collection("model.in")[0]

    validation_X, validation_Y = dataset.validation
    out = sess.run(model.output, feed_dict={model.input: validation_X})
    beta_eval = sess.run(beta)[0]

    #print(np.argsort(beta_eval)[-10:])
    #print(np.sort(beta_eval)[-10:])

    print(data.informative_feats)

    score = accuracy_score(y_true=validation_Y, y_pred=np.round(out))
    print("Accuracy score on validation: {:.2f}".format(score))

    train_X, train_Y = dataset.train
    out = sess.run(model.output, feed_dict={model.input: train_X})
    print(out.shape)
    train_score = accuracy_score(y_true=train_Y, y_pred=np.round(out))
    sess.close()
    print("Accuracy score on train: {:.2f}".format(train_score))

    sess.close()

