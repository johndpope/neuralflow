import tensorflow as tf


def norm(x, norm_type: str = "l2"):
    x_flattened = tf.reshape(x, [-1])
    norm_type = norm_type.lower()
    if norm_type == "l2":

        return tf.sqrt(tf.reduce_sum(x_flattened ** 2))

    elif norm_type == "l1":

        return tf.reduce_max(x_flattened)
    else:
        raise AttributeError("Unsupported norm type: {}, choose from 'l1, 'l2''".format(norm_type))
