import abc
import tensorflow as tf
import os
from neuralflow.models.Function import Function
from typing import List


class Model:
    __metaclass__ = abc.ABCMeta

    def __init__(self, input, output, n_in, n_out, trainables: List[tf.Variable] = ()):
        self.__input = input
        self.__output = output
        self.__n_in = n_in
        self.__n_out = n_out
        self.__trainables = trainables
        self.__meta_graph_saved = False
        self.__saver = None

    @property
    def input(self):
        """:returns the placeholder for the inputs of the model"""
        return self.__input

    @property
    def output(self):
        """:returns the placeholder for the outputs of the model"""
        return self.__output

    @property
    def n_in(self):
        """:returns the dimension of the input"""
        return self.__n_in

    @property
    def n_out(self):
        """:returns the dimension of the output"""
        return self.__n_out

    @property
    def trainables(self):
        """:returns the list of trainable variables"""
        return self.__trainables

    def save(self, output_dir: str, name:str, session: tf.Session):  # FIXME
        """save the model to file"""
        prefix = output_dir + "/" + name
        if not self.__meta_graph_saved:
            self.__saver = tf.train.Saver(var_list=self.trainables)
            tf.add_to_collection("model.out".format(name), self.output)
            tf.add_to_collection("model.in".format(name), self.input)
            os.makedirs(output_dir, exist_ok=True)
            # Generates MetaGraphDef.
            self.__saver.export_meta_graph(prefix + ".meta")
            self.__meta_graph_saved = True
        self.__saver.save(session, prefix, write_meta_graph=False)

    @staticmethod
    def from_external_input(n_in: int, float_type="float32"):
        input_placeholder = tf.placeholder(dtype=float_type, shape=(None, n_in), name="ExternalInput")
        return Model(input=input_placeholder, output=input_placeholder, n_in=n_in, n_out=n_in, trainables=[])

    @staticmethod
    def from_fnc(model, fnc: Function):
        output = fnc.apply(model.output)
        return Model(input=model.input, output=output, n_in=model.n_in, n_out=fnc.n_out,
                     trainables=model.trainables + fnc.trainables)
