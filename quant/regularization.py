import tensorflow as tf
import math

from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops


##inspired by https://github.com/itayhubara/BinaryNet.tf/blob/master/nnUtils.py

def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            x=tf.clip_by_value(x,-1,1)
            return tf.sign(x)

def ternarize(x):
    g = tf.get_default_graph()

    with ops.name_scope("Ternarized") as name:
        with g.gradient_override_map({"Round": "Identity", "Sign": "Identity"}):
            x = tf.round(10 * x)
            x = tf.clip_by_value(x, -1, 1)
            return x
