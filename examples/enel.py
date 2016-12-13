import numpy
import numpy as np
import tensorflow as tf
from neuralflow.enel.EnelDataset import EnelDataset
from neuralflow.math_utils import norm
from neuralflow.models.Model import Model
from neuralflow import BatchProducer
from neuralflow import GaussianInitialization

from neuralflow.neuralnets.FeedForwardNeuralNet import FeedForwardNeuralNet
from neuralflow.neuralnets.Layers import StandardLayerProducer, RBFLayerProducer
from neuralflow.optimization.Monitor import ScalarMonitor
from neuralflow.optimization.Criterion import ThresholdCriterion, MaxNoImproveCriterion, ImprovedValueCriterion
from neuralflow.neuralnets.ActivationFunction import SoftmaxActivationFunction, IdentityFunction, ReLUActivationFunction
from neuralflow.neuralnets.ActivationFunction import TanhActivationFunction
from neuralflow.neuralnets.LossFunction import CrossEntropy, SquaredError, MAE, EpsilonInsensitiveLoss
from neuralflow.optimization.IterativeTraining import IterativeTraining
from neuralflow.optimization.SupervisedOptimizationProblem import SupervisedOptimizationProblem
from neuralflow.optimization.GradientDescent import GradientDescent
from neuralflow.optimization.MultiTaskProblem import MultiTaskProblem
import time

from neuralflow.utils.LatexExporter import LatexExporter


def define_datasets(root_dir):
    scale_y = True
    # datasets = [EnelDataset(mat_file="/home/giulio/datasets/enel_mats/SUD_by_hour_complete.mat", seed=12,
    #                         scale_y=scale_y, name="SUD"),
    #             EnelDataset(mat_file="/home/giulio/datasets/enel_mats/CSUD_by_hour_complete.mat", seed=12,
    #                         scale_y=scale_y, name="CSUD")
    #             ]
    datasets = [EnelDataset(mat_file=root_dir+"/SUD_by_hour_preprocessed.mat", seed=12,
                            scale_y=scale_y, name="SUD")]

    # datasets = []
    # for i in range(24):
    #     datasets.append(EnelDataset(mat_file="/home/giulio/datasets/enel_mats/SUD_by_hour_preprocessed.mat", seed=12,
    #                                 scale_y=scale_y, name="SUD", hour=i))
    return datasets


def define_core_network(n_in: int, n_units: int):
    layer_prod_1 = StandardLayerProducer(n_units=n_units, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                         activation_fnc=TanhActivationFunction())

    net = FeedForwardNeuralNet(n_in, name="common")
    net.add_layer(layer_prod_1)
    return net


def define_task(core_net, dataset, name, loss_fnc):
    example = dataset.get_batch(1)
    n_in, n_out = example["input"].shape[1], example["output"].shape[1]
    print("\t {}-> n_in:{}, n_out:{}".format(dataset.name, n_in, n_out))

    input_layer = StandardLayerProducer(n_units=core_net.n_in,
                                        initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                        activation_fnc=TanhActivationFunction())

    input_net = FeedForwardNeuralNet(n_in=n_in,
                                     layer_producers=[input_layer], name="input_" + name)
    # out_layer_1 = RBFLayerProducer(n_units=100, initialization=GaussianInitialization(mean=0, std_dev=0.1))
    out_layer_2 = StandardLayerProducer(n_units=n_out, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                        activation_fnc=IdentityFunction())

    output_net = FeedForwardNeuralNet(n_in=core_net.n_out,
                                      layer_producers=[out_layer_2], name="output_" + name)

    model = Model.from_external_input(n_in=n_in)
    model = Model.from_fnc(model=model, fnc=input_net)
    model = Model.from_fnc(model=model, fnc=core_net)
    model = Model.from_fnc(model=model, fnc=output_net)

    batch_producer = dataset
    problem = SupervisedOptimizationProblem(model=model, loss_fnc=loss_fnc, batch_producer=batch_producer,
                                            batch_size=20)

    return problem, model


def attach_monitors(datasets, problems, models, training, multi_task_problem, optimizer):
    validation_monitors = []
    test_monitors = []
    train_monitors = []
    feed_dict_train = {}
    feed_dict_test = {}
    feed_dict_validation = {}

    for dataset, problem, model in zip(datasets, problems, models):
        mean = dataset.y_scaler.mean_
        scale = dataset.y_scaler.scale_

        def scale_placeholder(x):
            return (x * scale) + mean

        loss_monitor = ScalarMonitor(name=dataset.name + "_loss", variable=problem.objective_fnc_value)

        mae_loss = MAE(scale_placeholder)
        mae = mae_loss.value(model.output, problem.labels)
        mae_monitor = ScalarMonitor(name=dataset.name + "_MAE", variable=mae)

        validation_monitors.append(loss_monitor)
        validation_monitors.append(mae_monitor)
        train_monitors.append(mae_monitor)
        test_monitors.append(mae_monitor)

        validation_batch = dataset.get_validation()
        feed_dict_validation.update({model.input: validation_batch["input"],
                                     problem.labels: validation_batch["output"]})

        train_batch = dataset.get_train()
        feed_dict_train.update({model.input: train_batch["input"],
                                problem.labels: train_batch["output"]})

        test_batch = dataset.get_test()
        feed_dict_test.update({model.input: test_batch["input"],
                               problem.labels: test_batch["output"]})

    freq = 100
    grad_monitor = ScalarMonitor(name="grad_norm", variable=norm(optimizer.gradient, norm_type="l2"))
    training.add_monitors_and_criteria(monitors=[grad_monitor], freq=freq, name="batch")

    multi_task_loss_monitor = ScalarMonitor(name="loss", variable=multi_task_problem.objective_fnc_value)
    stopping_criterion = MaxNoImproveCriterion(monitor=multi_task_loss_monitor, max_no_improve=30, direction='<')
    saving_criterion = ImprovedValueCriterion(monitor=multi_task_loss_monitor, direction='<')

    training.add_monitors_and_criteria(monitors=train_monitors, freq=freq, name="train", feed_dict=feed_dict_train)
    training.add_monitors_and_criteria(monitors=validation_monitors + [multi_task_loss_monitor], freq=freq,
                                       name="validation", feed_dict=feed_dict_validation,
                                       stopping_criteria=[stopping_criterion], saving_criteria=[saving_criterion])
    training.add_monitors_and_criteria(monitors=test_monitors, freq=freq, name="test", feed_dict=feed_dict_test)
    training.add_monitors_and_criteria(monitors=[grad_monitor], freq=freq, name="batch")


def train_instance(datasets, parameters):
    tf.reset_default_graph()
    print("\tBeginning training instance...")
    print("\t" + str(parameters))

    output_dir = parameters["out_dir"] + str(parameters["id"])
    core_net = define_core_network(n_in=parameters["n_in"], n_units=parameters["n_hidden"])

    # objective function
    loss_fnc = EpsilonInsensitiveLoss(parameters["eps"])

    problems = []
    models = []
    for dataset in datasets:
        multi_task_problem, model = define_task(core_net=core_net, dataset=dataset, name=dataset.name,
                                                loss_fnc=loss_fnc)
        problems.append(multi_task_problem)
        models.append(model)

    multi_task_problem = MultiTaskProblem(problems=problems)
    # optimizer
    optimizer = GradientDescent(lr=0.01, max_norm=0.5, problem=multi_task_problem)

    # training
    training = IterativeTraining(max_it=10 ** 6, optimizer=optimizer, problem=multi_task_problem, output_dir=output_dir)

    attach_monitors(datasets=datasets, models=models, problems=problems, training=training,
                    multi_task_problem=multi_task_problem, optimizer=optimizer)

    sess = tf.Session()
    training.train(sess)
    error = test_instance(datasets, output_dir, sess)
    sess.close()

    return error


def test_instance(datasets, output_dir, sess):
    print("\tPerforming predictions...")
    errors = []
    for i, dataset in enumerate(datasets):
        new_saver = tf.train.import_meta_graph(output_dir + '/best_checkpoint_{}.meta'.format(i))
        new_saver.restore(sess, output_dir + '/best_checkpoint_{}'.format(i))

        net_out = tf.get_collection("model_{}.out".format(i))[0]  # XXX
        net_in = tf.get_collection("model_{}.in".format(i))[0]

        predictions = sess.run(net_out, feed_dict={net_in: dataset.get_test()["input"]})
        labels = dataset.get_test()["output"]

        labels = dataset.y_scaler.inverse_transform(labels)
        predictions = dataset.y_scaler.inverse_transform(predictions)

        error = np.mean(abs(predictions - labels))
        errors.append(error)

    cumulative_error = np.mean(np.array(errors))
    print("\tMAE: {}".format(cumulative_error))
    return cumulative_error


def export_results(result_list, output_dir):

    exp = LatexExporter(filename=output_dir+"results")
    exp.export(result_list)


if __name__ == "__main__":
    root_dir = "/home/galvan/"
    output_dir = root_dir+ "tensorBoards/enel/"
    dataset_dir = root_dir + "datasets/enel_mats/"
    datasets = define_datasets(dataset_dir)

    result_list = []

    id = 0
    for eps in [0.1, 0.005, 0.001]:
        for n_in in [25, 50, 100, 200, 300]:
            for n_hidden in [25, 50, 100, 200, 300]:
                parameters = {
                    "id": id,
                    "eps": eps,
                    "n_in": n_in,
                    "n_hidden": n_hidden,
                    "out_dir": output_dir
                }
                id += 1
                print("Beginning instance {}...".format(id))
                error = train_instance(datasets=datasets, parameters=parameters)
                result_list.append({
                    "eps": eps,
                    "n_in": n_in,
                    "n_hidden": n_hidden,
                    "MAE": "{:.2f}".format(error)})
    export_results(result_list, output_dir)
