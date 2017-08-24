import tensorflow as tf


def norm(x, norm_type: str = "l2"):
    # x_flattened = tf.reshape(x, [-1])
    # print(x_flattened)
    # x_flattened = x
    norm_type = norm_type.lower()
    if norm_type == "l2":

        return tf.sqrt(tf.reduce_sum(x ** 2))

    elif norm_type == "l1":

        return tf.reduce_max(x)
    else:
        raise AttributeError("Unsupported norm type: {}, choose from 'l1, 'l2''".format(norm_type))


def merge_all_tensors(tensors_tf: list):
    vars = [tf.reshape(g, [-1]) for g in tensors_tf]
    v = tf.concat(vars, 0)
    return v
