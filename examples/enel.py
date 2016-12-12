import numpy
import numpy as np
import tensorflow as tf
from neuralflow.enel.EnelDataset import EnelDataset
from neuralflow.math_utils import norm
from neuralflow.models.Model import Model
from neuralflow import BatchProducer
from neuralflow import GaussianInitialization

from neuralflow.neuralnets.FeedForwardNeuralNet import FeedForwardNeuralNet
from neuralnets.Layers import StandardLayerProducer, RBFLayerProducer
from neuralflow.optimization.Monitor import ScalarMonitor
from neuralflow.optimization.Criterion import ThresholdCriterion, MaxNoImproveCriterion
from neuralflow.neuralnets.ActivationFunction import SoftmaxActivationFunction, IdentityFunction, ReLUActivationFunction
from neuralflow.neuralnets.ActivationFunction import TanhActivationFunction
from neuralflow.neuralnets.LossFunction import CrossEntropy, SquaredError, MAE, EpsilonInsensitiveLoss
from neuralflow.optimization.IterativeTraining import IterativeTraining
from neuralflow.optimization.SupervisedOptimizationProblem import SupervisedOptimizationProblem
from neuralflow.optimization.GradientDescent import GradientDescent
from optimization.MultiTaskProblem import MultiTaskProblem


def define_core_network(n_in):
    layer_prod_1 = StandardLayerProducer(n_units=100, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                         activation_fnc=TanhActivationFunction())

    net = FeedForwardNeuralNet(n_in, name="common")
    net.add_layer(layer_prod_1)
    return net


def define_task(core_net, dataset, name):
    example = dataset.get_batch(1)
    n_in, n_out = example["input"].shape[1], example["output"].shape[1]

    input_layer = StandardLayerProducer(n_units=core_net.n_in,
                                        initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                        activation_fnc=TanhActivationFunction())

    input_net = FeedForwardNeuralNet(n_in=n_in,
                                     layer_producers=[input_layer], name="input_" + name)
    out_layer_1 = RBFLayerProducer(n_units=100, initialization=GaussianInitialization(mean=0, std_dev=0.1))
    out_layer_2 = StandardLayerProducer(n_units=n_out, initialization=GaussianInitialization(mean=0, std_dev=0.1),
                                        activation_fnc=IdentityFunction())

    output_net = FeedForwardNeuralNet(n_in=core_net.n_out,
                                      layer_producers=[out_layer_2], name="output_" + name)

    model = Model.from_external_input(n_in=n_in)
    model = Model.from_fnc(model=model, fnc=input_net)
    model = Model.from_fnc(model=model, fnc=core_net)
    model = Model.from_fnc(model=model, fnc=output_net)

    print("n_in:{}, n_out:{}".format(n_in, n_out))

    batch_producer = dataset
    # objective function
    loss_fnc = EpsilonInsensitiveLoss(0.05)

    problem = SupervisedOptimizationProblem(model=model, loss_fnc=loss_fnc, batch_producer=batch_producer,
                                            batch_size=20)

    return problem, model


if __name__ == "__main__":

    output_dir = "/home/giulio/tensorBoards/"
    scale_y = True
    datasets = [EnelDataset(mat_file="/home/giulio/datasets/enel_mats/SUD_by_hour_complete.mat", seed=12,
                            scale_y=scale_y, name="SUD"),
                EnelDataset(mat_file="/home/giulio/datasets/enel_mats/CSUD_by_hour_complete.mat", seed=12,
                            scale_y=scale_y, name="CSUD")
                ]

    # datasets = [EnelDataset(mat_file="/home/giulio/datasets/enel_mats/SUD_by_hour_complete.mat", seed=12,
    #                         scale_y=scale_y, name="SUD")]

    core_net = define_core_network(n_in=200)

    problems = []
    models = []
    for dataset in datasets:
        multi_task_problem, model = define_task(core_net=core_net, dataset=dataset, name=dataset.name)
        problems.append(multi_task_problem)
        models.append(model)

    multi_task_problem = MultiTaskProblem(problems=problems)
    # optimizer
    optimizer = GradientDescent(lr=0.01, max_norm=0.5, problem=multi_task_problem)

    # training
    training = IterativeTraining(max_it=10 ** 6, optimizer=optimizer, problem=multi_task_problem, output_dir=output_dir)

    validation_monitors = []
    test_monitors = []
    train_monitors = []
    feed_dict_train = {}
    feed_dict_test = {}
    feed_dict_validation = {}

    for dataset, problem, model in zip(datasets, problems, models):

        if scale_y:
            mean = dataset.y_scaler.mean_
            scale = dataset.y_scaler.scale_
            # print("Scale: {:.2f}".format(scale))
            # print("Mean: {:.2f}".format(mean))


        def scale_placeholder(x):
            if scale_y:
                return (x * scale) + mean
            else:
                return x


        loss_monitor = ScalarMonitor(name=dataset.name + "_loss", variable=problem.objective_fnc_value)

        mae_loss = MAE(scale_placeholder)
        mae = mae_loss.value(model.output, problem.labels)
        mae_monitor = ScalarMonitor(name=dataset.name + "_MAE", variable=mae)

        freq = 100

        validation_monitors.append(loss_monitor)
        validation_monitors.append(mae_monitor)
        train_monitors.append(mae_monitor)
        test_monitors.append(mae_monitor)

        validation_batch = dataset.get_validation()
        feed_dict_validation.update({model.input: validation_batch["input"],
                                     problem.labels: validation_batch["output"]})
        # training.add_monitors_and_criteria(monitors=[loss_monitor, mae_monitor], freq=freq, name="validation_" + dataset.name,
        #                                    feed_dict={model.input: validation_batch["input"],
        #                                               problem.labels: validation_batch["output"]})

        train_batch = dataset.get_train()
        feed_dict_train.update({model.input: train_batch["input"],
                                problem.labels: train_batch["output"]})
        # training.add_monitors_and_criteria(monitors=[mae_monitor], freq=freq, name="train_" + dataset.name,
        #                                    feed_dict={model.input: train_batch["input"],
        #                                               problem.labels: train_batch["output"]})

        test_batch = dataset.get_test()
        feed_dict_test.update({model.input: test_batch["input"],
                               problem.labels: test_batch["output"]})
        # training.add_monitors_and_criteria(monitors=[mae_monitor], freq=freq, name="test_" + dataset.name,
        #                                    feed_dict={model.input: test_batch["input"],
        #                                               problem.labels: test_batch["output"]})

    grad_monitor = ScalarMonitor(name="grad_norm", variable=norm(optimizer.gradient, norm_type="l2"))
    training.add_monitors_and_criteria(monitors=[grad_monitor], freq=101, name="batch")

    freq = 100
    multi_task_loss_monitor = ScalarMonitor(name="loss", variable=multi_task_problem.objective_fnc_value)
    stopping_criterion = MaxNoImproveCriterion(monitor=multi_task_loss_monitor, max_no_improve=50, direction='<')

    training.add_monitors_and_criteria(monitors=train_monitors, freq=freq, name="train", feed_dict=feed_dict_train)
    training.add_monitors_and_criteria(monitors=validation_monitors + [multi_task_loss_monitor], freq=freq,
                                       name="validation", feed_dict=feed_dict_validation,
                                       stopping_criteria=[stopping_criterion])
    training.add_monitors_and_criteria(monitors=test_monitors, freq=freq, name="test", feed_dict=feed_dict_test)
    training.add_monitors_and_criteria(monitors=[grad_monitor], freq=101, name="batch")

    sess = tf.Session()
    training.train(sess)
    sess.close()
