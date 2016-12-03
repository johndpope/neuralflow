
from __future__ import absolute_import

from tensorflow.contrib.metrics.python.ops.metric_ops import _broadcast_weights
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops
import tensorflow as tf
from tensorflow.contrib.framework import tensor_util


def _remove_squeezable_dimensions(predictions, labels, weights):
    """Squeeze last dim if needed.
    Squeezes `predictions` and `labels` if their rank differs by 1.
    Squeezes `weights` if its rank is 1 more than the new rank of `predictions`
    This will use static shape if available. Otherwise, it will add graph
    operations, which could result in a performance hit.
    Args:
      predictions: Predicted values, a `Tensor` of arbitrary dimensions.
      labels: Label values, a `Tensor` whose dimensions match `predictions`.
      weights: optional `weights` tensor. It will be squeezed if its rank is 1
        more than the new rank of `predictions`
    Returns:
      Tuple of `predictions`, `labels` and `weights`, possibly with the last
      dimension squeezed.
    """
    # predictions, labels = tensor_util.remove_squeezable_dimensions( # bhooo
    #     predictions, labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    if weights is not None:
        predictions_shape = predictions.get_shape()
        predictions_rank = predictions_shape.ndims
        weights_shape = weights.get_shape()
        weights_rank = weights_shape.ndims

        if (predictions_rank is not None) and (weights_rank is not None):
            # Use static rank.
            if weights_rank - predictions_rank == 1:
                weights = array_ops.squeeze(weights, [-1])
        elif (weights_rank is None) or (
                weights_shape.dims[-1].is_compatible_with(1)):
            # Use dynamic rank
            weights = tf.control_flow_ops.cond(
                math_ops.equal(array_ops.rank(weights),
                               math_ops.add(array_ops.rank(predictions), 1)),
                lambda: array_ops.squeeze(weights, [-1]),
                lambda: weights)
    return predictions, labels, weights


def _tp_fn_tn_fp(predictions, labels, thresholds, weights=None):
    predictions, labels, weights = _remove_squeezable_dimensions(
        predictions, labels, weights)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    num_thresholds = len(thresholds)

    # Reshape predictions and labels.
    predictions_2d = array_ops.reshape(predictions, [-1, 1])
    labels_2d = array_ops.reshape(
        math_ops.cast(labels, dtype=tf.bool), [1, -1])

    # Use static shape if known.
    num_predictions = predictions_2d.get_shape().as_list()[0]

    # Otherwise use dynamic shape.
    if num_predictions is None:
        num_predictions = array_ops.shape(predictions_2d)[0]
    thresh_tiled = array_ops.tile(
        array_ops.expand_dims(array_ops.constant(thresholds), [1]),
        array_ops.pack([1, num_predictions]))

    # Tile the predictions after thresholding them across different thresholds.
    pred_is_pos = math_ops.greater(
        array_ops.tile(array_ops.transpose(predictions_2d), [num_thresholds, 1]),
        thresh_tiled)
    pred_is_neg = math_ops.logical_not(pred_is_pos)

    # Tile labels by number of thresholds
    label_is_pos = array_ops.tile(labels_2d, [num_thresholds, 1])
    label_is_neg = math_ops.logical_not(label_is_pos)

    is_true_positive = math_ops.to_float(
        math_ops.logical_and(label_is_pos, pred_is_pos))
    is_false_negative = math_ops.to_float(
        math_ops.logical_and(label_is_pos, pred_is_neg))
    is_false_positive = math_ops.to_float(
        math_ops.logical_and(label_is_neg, pred_is_pos))
    is_true_negative = math_ops.to_float(
        math_ops.logical_and(label_is_neg, pred_is_neg))

    if weights is not None:
        weights = math_ops.to_float(weights)
        weights_tiled = array_ops.tile(array_ops.reshape(_broadcast_weights(
            weights, predictions), [1, -1]), [num_thresholds, 1])

        thresh_tiled.get_shape().assert_is_compatible_with(
            weights_tiled.get_shape())
        is_true_positive *= weights_tiled
        is_false_negative *= weights_tiled
        is_false_positive *= weights_tiled
        is_true_negative *= weights_tiled

    true_positives = math_ops.reduce_sum(is_true_positive, 1)
    false_negatives = math_ops.reduce_sum(is_false_negative, 1)
    true_negatives = math_ops.reduce_sum(is_true_negative, 1)
    false_positives = math_ops.reduce_sum(is_false_positive, 1)

    return (true_positives, false_negatives, true_negatives, false_positives)


def auc(predictions, labels, weights=None, num_thresholds=200,
        metrics_collections=None, curve='ROC', name=None):
    with variable_scope.variable_scope(name, 'auc', [predictions, labels]):
        if curve != 'ROC' and curve != 'PR':
            raise ValueError('curve must be either ROC or PR, %s unknown' %
                             (curve))
        kepsilon = 1e-7  # to account for floating point imprecisions
        thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                      for i in range(num_thresholds - 2)]
        thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

        tp, fn, tn, fp = _tp_fn_tn_fp(predictions, labels, thresholds, weights)

        # Add epsilons to avoid dividing by 0.
        epsilon = 1.0e-6
        assert array_ops.squeeze(fp).get_shape().as_list()[0] == num_thresholds

        def compute_auc(tp, fn, tn, fp, name):
            """Computes the roc-auc or pr-auc based on confusion counts."""
            recall = math_ops.div(tp + epsilon, tp + fn + epsilon)
            if curve == 'ROC':
                fp_rate = math_ops.div(fp, fp + tn + epsilon)
                x = fp_rate
                y = recall
            else:  # curve == 'PR'.
                precision = math_ops.div(tp + epsilon, tp + fp + epsilon)
                x = recall
                y = precision
            return math_ops.reduce_sum(math_ops.mul(
                x[:num_thresholds - 1] - x[1:],
                (y[:num_thresholds - 1] + y[1:]) / 2.), name=name)

        # sum up the areas of all the trapeziums
        auc = compute_auc(tp, fn, tn, fp, 'value')

        # update_op = compute_auc(
        #     tp_update_op, fn_update_op, tn_update_op, fp_update_op, 'update_op')

        if metrics_collections:
            ops.add_to_collections(metrics_collections, auc)

        return auc
