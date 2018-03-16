import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib import rnn

import sys
import timeit
from tensorflow.python.ops import *

#-------------------------------------------------------------------------
# Dense layer methods
#-------------------------------------------------------------------------
def Circ_matmul(X, Wz, bsize):
	""" Compute matrix product Y = X*W
			Where X is IxM (stack of I 1xM vectors)
						W is MxN (single weight matrix)
						Y is IxN (stack of I 1xN vectors)

			W is composed of BxB circulant blocks, M and N divide B.
			The matmul is performed without constructing W explicitly

			X  = (I, 1, M)     -> (I, M/B,   1, B)
			Wz = (M/B, N/B, B) -> (1, M/B, N/B, B)
			Yz = (I, M/B, N/B, B) -> (I, N/B, B)
			Y  = (I, N)
	"""

	sx = X.get_shape().as_list()
	sW = Wz.get_shape().as_list()
	print(sx, sW)
	assert(len(sx) <= 2)
	assert(len(sW) == 3)
	M, N = sx[-1], sW[1]*bsize
	print("M, N", M, N)
	out_shape = [N] if len(sx) == 1 else [-1,N]

	Xz = tf.reshape(X,  [-1, M/bsize, 1, bsize])
	Wz = tf.reshape(Wz, [ 1, M/bsize, N/bsize, bsize])

	FXz = tf.ifft(tf.complex(Xz, tf.zeros_like(Xz)))
	FWz = tf.fft(tf.complex(Wz, tf.zeros_like(Wz)))
	FYz = tf.reduce_sum(FXz*FWz, axis=1)
	Yz = tf.real( tf.fft(FYz) )
	#Yz = tf.fft(FYz)
	#Cz = tf.imag(Yz)
	#Cz = tf.Print(Cz, [Cz], message="dense Cz", first_n=2, summarize=10)
	#Yz = tf.real(Yz)
	#Yz = tf.Print(Yz, [Yz], message="dense Yz", first_n=2, summarize=10)
	#Yz = Yz + Cz

	Y  = tf.reshape(Yz, out_shape)
	return Y

class real_CirRNNCell(rnn_cell_impl.BasicRNNCell):
	def __init__(self, num_units, block_size, activation=None, reuse=None, name=None):
		super(rnn_cell_impl.BasicRNNCell, self).__init__(_reuse=reuse, name=name)

		# Inputs must be 2-dimensional.
		# self.input_spec = base_layer.InputSpec(ndim=2)
		self.block_size = block_size
		self._num_units = num_units
		self._activation = activation or math_ops.tanh

	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units

	def build(self, inputs_shape):
		B = self.block_size
		if inputs_shape[1].value is None:
			raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
					   % inputs_shape)

		input_depth = inputs_shape[1].value
		self._kernel = self.add_variable(
			"kernel",
			shape=[(input_depth + self._num_units)/B, self._num_units/B, B])
		self._bias = self.add_variable(
			"bias",
			shape=[self._num_units],
			initializer=init_ops.zeros_initializer(dtype=self.dtype))

		self.built = True

	def call(self, inputs, state):
		B = self.block_size
		gate_inputs = Circ_matmul(
			array_ops.concat([inputs, state], 1), self._kernel, B)
		gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
		output = self._activation(gate_inputs)
		return output, output

def Circ_dense(inputs, units, block_size,
							 activation=None,
							 use_bias=True,
							 kernel_initializer=None,
							 bias_initializer=tf.zeros_initializer(),
							 trainable=True,
							 name=None,
							 **kwargs):
	""" inputs = tensor, shaped (I, M), M must divide block_size
			units  = integer, must be divide block_size
			output = tensor, shaped (I, units)

			Creates weight tensor shaped (B, M/B, units/B)
	"""
	B = block_size
	input_shape = inputs.get_shape().as_list()
	assert(input_shape[-1] % B == 0)
	assert(units % B == 0)

	with tf.variable_scope(name or "Circ_dense"):
		Wz = tf.get_variable("weights_z",
												 shape=(input_shape[-1]/B, units/B, B),
												 dtype=inputs.dtype,
												 initializer=kernel_initializer,
												 trainable=trainable)

		outputs = Circ_matmul(inputs, Wz, B)

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
def Circ_conv2d(X, Wz, ksize, bsize, padding="SAME"):
	""" Compute Y = conv2d(X, W)
			X is (I, R, C, M)
			W is (K, K, M, N)
			Y is (I, R, C, N)

			W(i,j,:,:) is block-circulant

			Xi = (I, R, C, K*K*M)   -> (I*R*C, K*K*M/B,   1, B)
			Wz = (K*K, M/B, N/B, B) -> (    1, K*K*M/B, N/B, B)
			Yz = (I*R*C, K*K*M/B, N/B, B) -> (I*R*C, N/B, B)
	"""
	sX = X.get_shape().as_list()
	sW = Wz.get_shape().as_list()
	assert(len(sX) == 4)
	assert(len(sW) == 4)
	assert(len(ksize) == 2)
	R, C, M, N = sX[1], sX[2], sX[3], sW[2]*bsize
	K1, K2 = ksize[0], ksize[1]

	Xi = tf.extract_image_patches(X, [1,K1,K2,1], strides=[1,1,1,1],
															 rates=[1,1,1,1], padding=padding)
	sXi = Xi.get_shape().as_list()
	Rn, Cn = sXi[1], sXi[2]
	Xz = tf.reshape(Xi, [-1, K1*K2*M/bsize, 1, bsize])
	Wz = tf.reshape(Wz,  [1, K1*K2*M/bsize, N/bsize, bsize])

	FXz = tf.ifft(tf.complex(Xz, tf.zeros_like(Xz)))
	FWz = tf.fft (tf.complex(Wz, tf.zeros_like(Wz)))
	FYz = tf.einsum('abcd,ebfd->afd', FXz, FWz)
	Yz = tf.real( tf.fft(FYz) )
	#Yz = tf.fft(FYz)
	#Cz = tf.imag(Yz)
	#Cz = tf.Print(Cz, [Cz], message="conv Cz", first_n=2, summarize=10)
	#Yz = tf.real(Yz)
	#Yz = tf.Print(Yz, [Yz], message="conv Yz", first_n=2, summarize=10)
	#Yz = Yz + Cz

	Y = tf.reshape(Yz, [-1, Rn, Cn, N])
	return Y

def Circ_conv_layer(inputs, filter_shape, units, block_size,
										padding="SAME",
										activation=None,
										use_bias=True,
										kernel_initializer=None,
										bias_initializer=tf.zeros_initializer(),
										trainable=True,
										name=None,
										**kwargs):
	""" inputs = tensor, shaped (I, R, C, M), M must divide block_size
			units  = integer, must be divide block_size
			output = tensor, shaped (I, R', C', units)

			Creates weight tensor shaped (B, fsize, M/B, units/B)
	"""
	B = block_size
	fsize = filter_shape[0]*filter_shape[1]
	input_shape = inputs.get_shape().as_list()
	assert(input_shape[-1] % B == 0)
	assert(units % B == 0)

	with tf.variable_scope(name or "BH_conv"):
		Wz = tf.get_variable("weights_z",
												 shape=(fsize, input_shape[-1]/B, units/B, B),
												 dtype=inputs.dtype,
												 initializer=kernel_initializer,
												 trainable=trainable)

		outputs = Circ_conv2d(inputs, Wz, filter_shape, B)

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
