import numpy as np
import tensorflow as tf
import hadamard as hd
import circulant as cr
import transforms
from tensorflow.python.layers.base import _unique_layer_name
from pkg_resources import parse_version

import sys
import timeit

WEIGHTS_IN_HADAMARD_BASIS = True
print("\nhadamard_tf.py: WEIGHTS_IN_HADAMARD_BASIS = %s\n" % str(WEIGHTS_IN_HADAMARD_BASIS))

#-------------------------------------------------------------------------
# BH_matmul (Ryan)
#-------------------------------------------------------------------------
def BH_matmul(X, Wz, block_size, transform="None"):
    B = block_size
    input_shape = X.get_shape().as_list()
    units = Wz.get_shape().as_list()[-1]
    if transform == "Fourier":
        print("BH_dense: Circ bsize=%d" % B)
        idx = cr.block_circ_idx(input_shape[-1], units, B)
    elif transform == "Hadamard":
        print("BH_dense: Hadamard bsize=%d" % B)
        if not WEIGHTS_IN_HADAMARD_BASIS:
            H = tf.constant(hd.hadamard(B), dtype=tf.float32)
            Wz = tf.reshape(Wz, [-1, B])
            Wz = tf.matmul(Wz, H)
            Wz = tf.reshape(Wz, [input_shape[-1]*units/B, 1])
        idx = hd.block_hadamard_idx(input_shape[-1], units, B)
    else:
        print("BH_dense: Invalid transform")
        assert(0)

    idx = tf.constant(idx, tf.int32)
    W = tf.gather(tf.reshape(Wz, [-1]), idx)
    outputs = tf.matmul(X, W)
    return outputs


#-------------------------------------------------------------------------
# Dense layer methods
#-------------------------------------------------------------------------
def BH_dense(inputs, units, block_size,
             transform="None",
             use_scaling=False,
             activation=None,
             use_bias=True,
             kernel_weights=None,
             kernel_initializer=None,
             bias_initializer=tf.zeros_initializer(),
             trainable=True,
             name=None,
             **kwargs):
  """ inputs = tensor, shaped (I, M), M must divide block_size
      units  = integer, must be divide block_size
      output = tensor, shaped (I, units)

      Creates weight tensor shaped (M/B, units/B, B)
  """
  B = block_size
  input_shape = inputs.get_shape().as_list()
  assert(input_shape[-1] % B == 0)
  assert(units % B == 0)

  # The name will be unique in the current (graph, variable_scope) but
  # different tf.name_scope() can share the same name
  if name is None:
    if parse_version(tf.__version__) >= parse_version('1.5.0-rc0'):
      name_scope = tf.get_variable_scope().original_name_scope
      name = _unique_layer_name("BH_dense", namespace=name_scope)
    else:
      name = _unique_layer_name("BH_dense")

  with tf.variable_scope(tf.get_variable_scope()):
    # kernel_weights lets the user pass in weights, otherwise we create weights
    if kernel_weights is not None:
      Wz = kernel_weights
    else:
      Wz = tf.get_variable("weights_z",
                           shape=(input_shape[-1]*units/B, 1),
                           dtype=inputs.dtype,
                           initializer=kernel_initializer,
                           trainable=trainable)

    if transform == "Fourier":
      print("BH_dense: Circ bsize=%d" % B)
      idx = cr.block_circ_idx(input_shape[-1], units, B)
    elif transform == "Hadamard":
      print("BH_dense: Hadamard bsize=%d" % B)
      if not WEIGHTS_IN_HADAMARD_BASIS:
        H = tf.constant(hd.hadamard(B), dtype=tf.float32)
        Wz = tf.reshape(Wz, [-1, B])
        Wz = tf.matmul(Wz, H)
        Wz = tf.reshape(Wz, [input_shape[-1]*units/B, 1])
      idx = hd.block_hadamard_idx(input_shape[-1], units, B)
    else:
      print("BH_dense: Invalid transform")
      assert(0)

    if use_scaling:
      # FIXME: not supporting this right now
      assert(0)

    idx = tf.constant(idx, tf.int32)
    W = tf.gather(tf.reshape(Wz, [-1]), idx)
    outputs = tf.matmul(inputs, W)

    if use_bias:
      biases = tf.get_variable("biases",
                               shape=(units),
                               dtype=inputs.dtype,
                               initializer=bias_initializer,
                               trainable=trainable)
      outputs = outputs + biases

    if activation:
      outputs = activation(outputs)
    return outputs

#-------------------------------------------------------------------------
# Conv layer methods
#-------------------------------------------------------------------------
def BH_conv_layer(inputs, filter_shape, units, block_size,
                  transform="None",
                  use_scaling=False,
                  strides=1,
                  padding="SAME",
                  activation=None,
                  use_bias=True,
                  kernel_weights=None,
                  kernel_initializer=None,
                  bias_initializer=tf.zeros_initializer(),
                  trainable=True,
                  name=None,
                  **kwargs):
  """ inputs = tensor, shaped (I, R, C, M), M must divide block_size
      units  = integer, must be divide block_size
      output = tensor, shaped (I, R', C', units)

      Creates weights tensor shaped (K, K, M, N/B)
  """
  B = block_size
  fshape = filter_shape
  input_shape = inputs.get_shape().as_list()
  assert(input_shape[-1] % B == 0)
  assert(units % B == 0)

  # The name will be unique in the current (graph, variable_scope) but
  # different tf.name_scope() can share the same name
  if name is None:
    if parse_version(tf.__version__) >= parse_version('1.5.0-rc0'):
      name_scope = tf.get_variable_scope().original_name_scope
      name = _unique_layer_name("BH_conv", namespace=name_scope)
    else:
      name = _unique_layer_name("BH_conv")

  with tf.variable_scope(name):
    # kernel_weights lets the user pass in weights, otherwise we create weights
    if kernel_weights is not None:
      Wz = kernel_weights
    else:
      Wz = tf.get_variable("weights_z",
                           shape=(fshape[0], fshape[1], input_shape[-1]*units/B),
                           dtype=inputs.dtype,
                           initializer=kernel_initializer,
                           trainable=trainable)

    #if use_scaling:
    #  Fz = tf.get_variable("scaling_z",
    #                       shape=(fsize, input_shape[-1]/B, units/B, B),
    #                       dtype=inputs.dtype,
    #                       initializer=tf.constant_initializer(1.),
    #                       trainable=trainable)
    #else:
    #  Fz = None

    if transform == "Fourier":
      print("BH_conv_layer: Circ bsize=%d" % B)
      idx = cr.block_circ_idx(input_shape[-1], units, B)
    elif transform == "Hadamard":
      print("BH_conv_layer: Hadamard bsize=%d" % B)
      if not WEIGHTS_IN_HADAMARD_BASIS:
        H = tf.constant(hd.hadamard(B), dtype=tf.float32)
        Wz = tf.reshape(Wz, [-1, B])
        Wz = tf.matmul(Wz, H)
        Wz = tf.reshape(Wz, [fshape[0], fshape[1], input_shape[-1]*units/B])
      idx = hd.block_hadamard_idx(input_shape[-1], units, B)
    else:
      print("BH_conv_layer: Invalid transform")
      assert(0)

    if use_scaling:
      # FIXME: not supporting this right now
      assert(0)

    idx = tf.constant(idx, tf.int32)
    W = tf.gather(Wz, idx, axis=2)

    outputs = tf.nn.conv2d(inputs, W,
                           strides=[1,strides,strides,1],
                           padding=padding)

    if use_bias:
      biases = tf.get_variable("biases",
                               shape=(units),
                               dtype=inputs.dtype,
                               initializer=bias_initializer,
                               trainable=trainable)
      outputs = outputs + biases

    if activation:
      outputs = activation(outputs)
    return outputs

#-------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------

if __name__ == "__main__":
  print("ALL TESTS PASSED!")
