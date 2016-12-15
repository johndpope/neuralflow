import tensorflow as tf
import numpy as np


class Estimator:
    @staticmethod
    def load_from_file(prefix_path, in_out_keys):
        g = tf.Graph()
        with g.as_default():
            new_saver = tf.train.import_meta_graph(prefix_path + 'best_checkpoint.meta')

            def init_op(sess): new_saver.restore(sess, prefix_path + 'best_checkpoint')
        return Estimator(g, in_out_keys, init_op)

    def __init__(self, graph: tf.Graph, in_out_keys, init_op):
        self.__graph = graph
        self.__graph.as_default()
        self.__init_op = init_op
        self.__in_out_keys = in_out_keys

    def predict(self, X: np.array) -> np.array:
        with self.__graph.as_default():
            sess = tf.Session()
            self.__init_op(sess)
            net_out = tf.get_collection(self.__in_out_keys["out"])[0]  # XXX
            net_in = tf.get_collection(self.__in_out_keys["in"])[0]
            predictions = sess.run(net_out, feed_dict={net_in: X})
            sess.close()
        return predictions

        # def __del__(self):
        #     self.__sess.close()
