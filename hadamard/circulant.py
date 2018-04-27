import numpy as np
import tensorflow as tf
import sys
import timeit

def block_circ_idx(rows, cols, B):
  assert(rows % B == 0)
  assert(cols % B == 0)
  #rows = int((rows+B-1)/B) * B
  #cols = int((cols+B-1)/B) * B
  # idx is the index within a block
  i = np.arange(0,B,1).reshape([1,B])
  j = np.arange(0,-B,-1).reshape([B,1])
  idx = i + j
  idx = (idx + B) % B
  # offset is the shared offset for each block
  tiled = np.tile(idx, [int(rows/B), int(cols/B)])
  offset = np.arange(0,rows*cols)
  i = (offset / cols) / B
  j = (offset % cols) / B
  offset = (i * cols + j * B).reshape([rows,cols])
  return tiled + offset

#-------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------
def is_circulant(W):
  """ Looks at the last 2 dimensions of W, checks that each
      W(...,:,:) is a circulant matrix by comapring columns """
  assert (W.shape[-2] == W.shape[-1])

  err = 0
  for i in range(W.shape[-2]):
    rot = np.concatenate((W[...,0,-i:], W[...,0,:-i]), axis=W.ndim-2)
    err += np.sum(np.square(W[...,i,:] - rot))

  err /= W.shape[-1]*W.shape[-2]
  assert(err < 1e-6)

def is_block_circulant(W, B):
  """ W is 2D matrix, B is block size """
  assert(W.shape[0] % B == 0)
  assert(W.shape[1] % B == 0)
  blocks = [ W[i:i+B,j:j+B]
              for j in range(0, W.shape[1], B)
                for i in range(0, W.shape[0], B) ]
  blocks = np.stack(np.array(blocks))
  is_circulant(blocks)

def test_circ_structure(K, M, N, B):
  # Our goal is to create a circulant 4D filter weight with shape [K, K, M, N]
  W_ = np.arange(0, K*K*M*N/B).reshape([K, K, M*N/B])
  W = tf.Variable(W_, dtype=tf.float32)

  idx = block_circ_idx(M, N, B)
  print(idx)
  idx = tf.constant(idx, dtype=tf.int32)
  C = tf.gather(W, idx, axis=2)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    W_ = W.eval()
    C_ = C.eval()

  #print("W_:")
  #print(W_)
  #print("C_:")
  #print(C_)

if __name__ == "__main__":
  # np test
  R = 8
  C = 12
  B = 4
  Wz = np.random.rand(R*C/B)
  idx = block_circ_idx(R, C, B)
  W = Wz[idx]

  #print(W)
  is_block_circulant(W, B)

  # tf test
  Wz = tf.random_uniform([R*C/B], -1, 1)
  W = tf.gather(Wz, idx)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    W_ = W.eval()

  #print(W_)
  is_block_circulant(W_, B)
  
  test_circ_structure(3, 8, 12, 4)

  print("ALL TESTS PASSED!")
