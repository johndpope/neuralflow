import tensorflow as tf
import numpy as np


class Estimator:
    @staticmethod
    def load_from_file(prefix_path, in_out_keys):

        def init_op():
            new_saver = tf.train.import_meta_graph(prefix_path + 'best_checkpoint.meta')
            config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
            )
            sess = tf.Session()#config=config)
            new_saver.restore(sess, prefix_path + 'best_checkpoint')
            return sess
        return Estimator(in_out_keys, init_op)

    def __init__(self, in_out_keys, init_op):
        self.__init_op = init_op
        self.__in_out_keys = in_out_keys

    def predict(self, X: np.array) -> np.array:
        tf.reset_default_graph()
        sess = self.__init_op()
        net_out = tf.get_collection(self.__in_out_keys["out"])[0]  # XXX
        net_in = tf.get_collection(self.__in_out_keys["in"])[0]
        predictions = sess.run(net_out, feed_dict={net_in: X})
        sess.close()
        sess = None

        g = tf.get_default_graph()
        keys = g.get_all_collection_keys()
        for key in keys:
            g.clear_collection(key)
        tf.reset_default_graph()
        return predictions

