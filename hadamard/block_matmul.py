import numpy as np
import sys

def block_diag_matmul(X, W_, bsize):
  """ Compute matrix product Y = X*W
      Where X is IxM (stack of I 1xM vectors)
            W is MxN (single weight matrix)
            Y is IxN (stack of I 1xN vectors)

      W is composed of BxB diagonal blocks, M and N divide B.
      W is passed as W_, a Bx(M/B)x(N/B) vector, each Bx1x1 vector
      is a diagonal of a block of W.
      The matmul is performed efficiently without constructing W.

      X  = (I, 1, M) -> (I, M/B, B) -> (B, I, M/B)
      X_ = (B, I, M/B)
      W_ = (B, M/B, N/B)    *1st dim match, 2nd dim broadcast, 3rd dim sum*
      Y_ = (B, I, N/B)
      Y  = (B, I, N/B) -> (I, N/B, B) -> (I, N)
  """
  assert(W_.ndim == 3)
  I, M, N = X.shape[0], X.shape[1], W_.shape[2]*bsize
  assert(W_.shape[0] == bsize)
  assert(W_.shape[1]*bsize == M)

  X_ = X.reshape(I, M/bsize, bsize)
  X_ = np.swapaxes(np.swapaxes(X_, 1,2), 0,1)
  Y_ = np.matmul(X_, W_)
  Y_ = np.swapaxes(np.swapaxes(Y_, 0,1), 1,2)
  Y = Y_.reshape(I, N)
  return Y

def block_diag_full(W_):
  """ Constructs W, MxN matrix composed of BxB diagonal blocks,
      given W_, Bx(M/B)x(N/B) stack of diagonal entries"""
  assert(W_.ndim == 3)
  bsize = W_.shape[0]
  full = np.concatenate([
            np.concatenate([ np.diag(W_[:,i,j]) for j in range(W_.shape[2]) ], axis=1)
         for i in range(W_.shape[1]) ], axis=0)
  return full

def block_diag_zip(W, bsize):
  """ Constructs W_, Bx(M/B)x(N/B) stack of diagonal entries,
      given W, MxN block diagonal matrix"""
  assert(W.ndim == 2)
  assert(W.shape[0] % bsize == 0)
  assert(W.shape[1] % bsize == 0)
  zipped = np.array([
              [ np.diag(W[i:i+bsize,j:j+bsize]) for j in range(0,W.shape[1],bsize) ]
           for i in range(0,W.shape[0],bsize) ])
  zipped = np.swapaxes(np.swapaxes(zipped, 1,2), 0,1)
  return zipped

if __name__ == "__main__":
  B = 2
  M = 6
  N = 10
  I = 3

  X  = np.random.randint(-9, 9, (I,M))
  W_ = np.random.randint(-9, 9, (B, M/B, N/B))
  W  = block_diag_full(W_)
  assert(np.sum(np.square(block_diag_zip(W,B) - W_)) < 1e-10)

  print("\nX:")
  print(X)
  print("\nW_:")
  print(W_)
  print("\nW:")
  print(W)

  Y = np.matmul(X,W)
  print("\nY:")
  print(Y)

  Yd = block_diag_matmul(X, W_, B)
  print("\nYd:")
  print(Yd)

  assert(np.sum(np.square(Yd-Y)) < 1e-10)
