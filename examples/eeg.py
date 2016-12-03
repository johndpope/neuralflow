import numpy as np
from Kaggle.MachineLearning.aggregate_by_id import aggregate_by_id
from Kaggle.Utils.SubmissionExporter import SubmissionExporter
# from SessionManager import SessionManager
from neuralflow.math_utils import norm
from neuralflow.models.Model import ExternalInputModel
from neuralflow import BatchProducer
from neuralflow import GaussianInitialization
from neuralflow import ValidationProducer
from neuralflow.neuralnets.FeedForwardNeuralNet import FeedForwardNeuralNet
from neuralnets.Layers import StandardLayerProducer
from neuralflow.optimization.Monitor import ScalarMonitor, RocMonitor
from neuralflow.optimization.Criterion import ThresholdCriterion, MaxNoImproveCriterion, ImprovedValueCriterion
from neuralflow.neuralnets.ActivationFunction import SoftmaxActivationFunction
from neuralflow.neuralnets.ActivationFunction import TanhActivationFunction
from neuralflow.neuralnets.LossFunction import CrossEntropy, HingeLoss
from neuralflow.optimization.IterativeTraining import IterativeTraining
from neuralflow.optimization.SupervisedOptimizationProblem import SupervisedOptimizationProblem
from neuralflow.optimization.GradientDescent import GradientDescent
from scipy.stats import rankdata
from sklearn import preprocessing
import pickle
import os
from Kaggle.convert_dataset_neural import pack_dataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler


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

        x_scaler = preprocessing.StandardScaler().fit(np.concatenate((x_train, x_validation, x_test)))
        # x_scaler = preprocessing.MinMaxScaler().fit(np.concatenate((x_train, x_validation, x_test)))

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


def define_problem(dataset, seed, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # train
    example = dataset.get_batch(20)
    # print(example)
    n_in, n_out = example["input"].shape[1], example["output"].shape[1]

    hidden_layer_prod_1 = StandardLayerProducer(n_units=500,
                                                initialization=GaussianInitialization(mean=0, std_dev=0.1, seed=seed),
                                                activation_fnc=TanhActivationFunction())

    hidden_layer_prod_2 = StandardLayerProducer(n_units=2000,
                                                initialization=GaussianInitialization(mean=0, std_dev=0.1, seed=seed),
                                                activation_fnc=TanhActivationFunction())

    hidden_layer_prod_3 = StandardLayerProducer(n_units=500,
                                                initialization=GaussianInitialization(mean=0, std_dev=0.1, seed=seed),
                                                activation_fnc=TanhActivationFunction())

    hidden_layer_prod_4 = StandardLayerProducer(n_units=100,
                                                initialization=GaussianInitialization(mean=0, std_dev=0.1, seed=seed),
                                                activation_fnc=TanhActivationFunction())
    output_layer_prod = StandardLayerProducer(n_units=n_out,
                                              initialization=GaussianInitialization(mean=0, std_dev=0.1, seed=seed),
                                              activation_fnc=SoftmaxActivationFunction(single_output=True))

    net = FeedForwardNeuralNet(input_model=ExternalInputModel(n_in=n_in))
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

    problem = SupervisedOptimizationProblem(model=net, loss_fnc=loss_fnc, batch_producer=batch_producer, batch_size=20)
    # optimizer
    optimizer = GradientDescent(lr=0.05, problem=problem)

    grad_monitor = ScalarMonitor(name="grad_norm", variable=norm(optimizer.gradient, norm_type="l2"))

    # training
    training = IterativeTraining(max_it=10 * 4, optimizer=optimizer, problem=problem, output_dir=output_dir)

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

    # add monitors and criteria on train-set
    # train_batch = dataset.get_train()
    # training.add_monitors_and_criteria(monitors=[tr_loss_monitor, tr_roc_monitor], freq=100, name="train",
    #                                    feed_dict={net.input: train_batch["input"],
    #                                               problem.labels: train_batch["output"]})

    return training, net


def process_patient(seed, pat_id):
    output_dir = "/home/giulio/tensorBoards/{}_{}/".format(seed, pat_id)
    data = pack_dataset(seed, dataset_type, root_dir, patient_ids=[pat_id])
    dataset = EegDataset(data=data, seed=seed)

    test_names = aggregate_by_id(data["test_segments"], data["test_names"])

    tf.reset_default_graph()
    sess = tf.Session()

    training, net = define_problem(dataset, seed, output_dir)
    training.train(sess)

    new_saver = tf.train.import_meta_graph(output_dir + 'best_checkpoint.meta')
    new_saver.restore(sess, output_dir + 'best_checkpoint')

    net_out = tf.get_collection("net.out")[0]  # XXX
    net_in = tf.get_collection("net.in")[0]

    test_predictions = sess.run(net_out, feed_dict={net_in: dataset.get_test()["input"]})
    test_predictions = aggregate_by_id(data["test_segments"], test_predictions, np.mean)

    validation_prediction = sess.run(net_out, feed_dict={net_in: dataset.get_validation()["input"]})
    validation_targets = dataset.get_validation()["output"]

    auc_splitted = roc_auc_score(y_true=validation_targets, y_score=validation_prediction)

    validation_prediction = aggregate_by_id(data["val_segments"], validation_prediction, np.mean)
    validation_targets = aggregate_by_id(data["val_segments"], validation_targets)

    auc_cumulative = roc_auc_score(y_true=validation_targets, y_score=validation_prediction)
    print("\tValidation-> auc_splitted:{:.3f}, auc_cumulative:{:.3f}".format(auc_splitted, auc_cumulative))

    sess.close()

    return validation_prediction, validation_targets, test_predictions, test_names, data["blacklist"]


# duplicate form Kaggle
def combine_predictions(*predictions):
    if not predictions:
        return None
    elif len(predictions) < 2:
        return np.array(predictions[0])
    else:
        acc = np.zeros_like(predictions[0])

        for p in predictions:
            ranking = rankdata(p, method="ordinal").astype(float)
            scaled_predictions = np.array(MinMaxScaler().fit_transform(ranking.reshape(-1, 1))).squeeze()
            acc += scaled_predictions
        return acc / len(predictions)


############## start seed looping #####################

import tensorflow as tf

# root_dir = "/home/giulio/"
root_dir = "/media/data-haydn/"
dataset_type = 'entropy_overlapping_bands'
seeds = range(5)
# seeds = [87]

test_predictions_runs = []
aucs = []
test_names = None
blacklist = None

for seed in seeds:
    print("Seed: {}".format(seed))
    test_names = np.array([])
    test_predictions_list = []
    validation_predictions_list = []
    validation_targets_list = []
    blacklist = []

    for pat_id in [1, 2, 3]:
        print("\tProcessing patient {}...".format(pat_id))
        validation_prediction_pat, validation_targets_pat, test_predictions_pat, test_names_pat, blacklist_pat = process_patient(
            seed, pat_id)
        test_names = np.concatenate((test_names, test_names_pat))
        blacklist += blacklist_pat
        validation_targets_list.append(validation_targets_pat)
        validation_predictions_list.append(validation_prediction_pat)
        test_predictions_list.append(test_predictions_pat)

    auc_all_pats = roc_auc_score(y_true=np.concatenate(validation_targets_list),
                                 y_score=np.concatenate(validation_predictions_list))
    print("Validation-> auc_cumulative_all_pats:{:.3f}".format(auc_all_pats))
    aucs.append(auc_all_pats)
    test_predictions_runs.append(np.concatenate(test_predictions_list))

print("Mean AUC:{:.4f}".format(np.mean(np.array(aucs))))
test_predictions = combine_predictions(*test_predictions_runs)
exporter = SubmissionExporter(output_filename="./submission_FFNN_final", append=False)
exporter.export_results(test_predictions, test_names, blacklist)



# sess.close()
#
# sess = tf.Session()
# new_saver = tf.train.import_meta_graph(output_dir + 'best_checkpoint.meta')
# new_saver.restore(sess, output_dir + 'best_checkpoint')
#
# net_out = tf.get_collection("net.out")[0]
# net_in = tf.get_collection("net.in")[0]
#
# out = sess.run(net_out, feed_dict={net_in: validation_batch["input"]})[0:5]
# print(out)
