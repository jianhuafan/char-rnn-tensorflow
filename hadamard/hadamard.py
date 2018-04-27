import numpy as np
import sys
import timeit
import block_matmul as bk
from transforms import fht, ifht, hadamard

def block_hadamard_idx(rows, cols, B):
  assert(rows % B == 0)
  assert(cols % B == 0)
  H = hadamard(B)*np.sqrt(B)
  idx = np.diag(np.matmul(H, np.arange(B)))
  idx = np.matmul(np.matmul(H, idx), H)
  idx = np.round(idx).astype(int) // B
  #print(idx)
  # offset is the shared offset for each block
  tiled = np.tile(idx, [int(rows/B), int(cols/B)])
  offset = np.arange(0,rows*cols)
  i = (offset / cols) / B
  j = (offset % cols) / B
  offset = (i * cols + j * B).reshape([rows,cols])
  return tiled + offset

#-------------------------------------------------------------------------
# Hadamard diagonalizable weight matrices (square)
#-------------------------------------------------------------------------
def hadamard_full(w, f=None):
  """ Explicitly computes W = inv(F)*inv(H) * diag(HFw) * HFx """
  assert(w.ndim == 1)
  N = w.shape[0]
  H = hadamard(N)

  if f is not None:
    w = f*w
  d = fht(w)
  W = np.dot(H, np.dot(np.diag(d), H))

  if f is not None:
    W = np.dot( np.dot(np.diag(1./f), W), np.diag(f) )
  return W

def hadamard_diag(W, f=None):
  """ Inverse of hadamard_full, converts N-by-N matrix W into diagonal w """
  assert(W.ndim == 2)
  N = W.shape[0]
  H = hadamard(N)

  if f is not None:
    W = np.dot( np.dot(np.diag(f), W), np.diag(1./f) )
  D = np.dot(H, np.dot(W, H) )

  d = np.diag(D)  # if W is not H-diagonalizable this will drop terms
  w = ifht(d)
  if f is not None:
    w = w/f
  return w

def Hmul(w, x, f=None):
  """
  Computes y = Wx,
  where W = inv(F)*inv(H) * diag(HFw) * HFx
        F = diag(f)
        H = Hadamard matrix
  """
  assert(w.ndim == 1)
  assert(x.ndim == 1)
  assert(w.shape[0] == x.shape[0])
  # Scale x and w by f
  if f is not None:
    x = f*x
    w = f*w
  # FHT on w and x, elementwise mul, then inverse FHT
  y = ifht(fht(w) * fht(x))
  # Undo scaling by f
  if f is not None:
    y = y/f
  return y

#-------------------------------------------------------------------------
# Block Hadamard
#-------------------------------------------------------------------------
def block_hadamard_full(Wz, Fz=None):
  """ Constructs W, MxN matrix of BxB Hadamard-diagonalizable blocks
      given Wz, (M/B)x(N/B)xB stack of diagonal entries. """
  assert(Wz.ndim == 3)
  bsize = Wz.shape[-1]
  full = np.concatenate([
            np.concatenate([
                hadamard_full( Wz[i,j,:], f=None if Fz is None else Fz[i,j,:] )
            for j in range(Wz.shape[1]) ], axis=1)
         for i in range(Wz.shape[0]) ], axis=0)
  return full

def block_hadamard_zip(W, bsize, Fz=None):
  """ Constructs Wz, (M/B)x(N/B)xB stack of diagonal entries,
      given W, MxN block Hadamard-diagonalizable matrix"""
  assert(W.ndim == 2)
  assert(W.shape[0] % bsize == 0)
  assert(W.shape[1] % bsize == 0)
  zipped = np.array([
    [ hadamard_diag( W[i:i+bsize,j:j+bsize], f=None if Fz is None else Fz[i/bsize,j/bsize,:] )
                for j in range(0,W.shape[1],bsize) ]
           for i in range(0,W.shape[0],bsize) ])
  return zipped

def block_Hmul(X, Wz, bsize, Fz=None):
  """ Compute matrix product Y = X*W
      Where X is IxM (stack of I 1xM vectors)
            W is MxN (single weight matrix)
            Y is IxN (stack of I 1xN vectors)

      W is composed of BxB Hadamard-diagonalizable blocks, M and N divide B.
      The matmul is performed without constructing W or Hadamard(B)

      X  = (I, 1, M) -> (I, M/B, 1, B)
      Wz = (M/B, N/B, B) -> (1, M/B, N/B, B)
      Fz = (1, M/B, N/B, B)
      Y_ = (I, M/B, N/B, B) -> (I, N/B, B)
      Y  = (I, N)
  """
  assert(Wz.ndim == 3)
  I, M, N = X.shape[0], X.shape[1], Wz.shape[1]*bsize
  assert(Wz.shape[-1] == bsize)
  assert(Wz.shape[0]*bsize == M)

  X_ =  X.reshape(I, M/bsize,       1, bsize)
  W_ = Wz.reshape(1, M/bsize, N/bsize, bsize)

  if Fz is None:
    Y_ = fht(np.sum(ifht(X_) * fht(W_), axis=1))  # (I, N/B, B)
  else:
    print("la F!")
    F_ = Fz.reshape(1, M/bsize, N/bsize, bsize)
    Y_ = np.sum( fht(ifht(X_/F_) * fht(W_*F_))*F_, axis=1 )

  Y = Y_.reshape(I, N)
  return Y

#-------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------
def is_hadamard_diag(W, f=None):
  w = hadamard_diag(W, f=f)
  M = hadamard_full(w, f=f)
  mse = np.sum(np.square(M-W))/W.size
  #print("is_hadamard_diag: %8.6f" % r)
  return mse < 1e-4

def is_block_hadamard_diag(W, B):
  assert(W.shape[0] % B == 0)
  assert(W.shape[1] % B == 0)
  for j in range(0, W.shape[1], B):
    for i in range(0, W.shape[0], B):
      is_hadamard_diag(W[i:i+B, j:j+B])

def test_fht(N):
  print("==== test_fht %d ====" % N)
  x = np.random.rand(N).astype(np.float32)
  H = hadamard(N)

  t0 = timeit.default_timer()
  y1 = np.dot(H,x)
  t1 = timeit.default_timer()
  y2 = fht(x)
  t2 = timeit.default_timer()
  mse = np.sum(np.square(y1-y2))/y1.size
  print("t1 : %6.3f\nt2 : %6.3f\nMSE: %8.6f" % (t1-t0, t2-t1, mse))
  assert(mse < 1e-4)

def test_Hmul(N, use_f=False):
  print("==== test_Hmul %d ====" % N)
  w = np.random.rand(N).astype(np.float32)
  f = np.random.rand(N).astype(np.float32) if use_f else None
  x = np.random.rand(N).astype(np.float32)

  W = hadamard_full(w, f=f)
  #print("\nweight vector:")
  #print(w)
  #print("\nhadamard_full:")
  #print(W)
  assert(is_hadamard_diag(W, f=f))

  y1 = np.dot(W,x)
  y2 = Hmul(w,x,f=f)
  mse = np.sum(np.square(y2-y1))/y1.size
  print("MSE: %8.6f" % mse)
  assert(mse < 1e-4)

def test_block_hadamard(Wz, Fz=None):
  W  = block_hadamard_full(Wz, Fz=Fz)
  Wn = block_hadamard_zip(W, B, Fz=Fz)
  assert(np.sum(np.square(Wn-Wz))/Wn.size < 1e-4)

def test_block_Hmul(X, Wz, bsize, Fz=None):
  print("==== test_block_Hmul %d ====" % bsize)
  W  = block_hadamard_full(Wz, Fz=Fz)
  Yf = np.matmul(X,W)
  Yd = block_Hmul(X,Wz,B, Fz=Fz)
  mse = np.sum(np.square(Yf-Yd)) / Yf.size
  print("MSE: %8.6f" % mse)
  assert(mse < 1e-3)

def test_hadamard_idx(M, N, B):
  Wz = np.random.rand(M*N/B)
  idx = block_hadamard_idx(M, N, B)
  W = Wz[idx]
  is_block_hadamard_diag(W,B)

#-------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------
if __name__ == "__main__":
  np.set_printoptions(precision=3)

  # Fast Hadamard test
  #for i in range(14,8,-1):
  #  test_fht(2**i)

  # Fast diagonal multiply test
  #for i in range(10,1,-1):
  #  test_Hmul(2**i, use_f=True)

  test_Hmul(2**10, use_f=False)
  test_Hmul(2**10, use_f=True)

  # This needs to fail
  W = np.random.rand(16,16).astype(np.float32)
  assert(not is_hadamard_diag(W))

  # Block tests
  B = 8
  M = 3*B
  N = 2*B
  I = 5
  X  = np.random.rand(I,M).astype(np.float32)
  Wz = np.random.rand(M/B, N/B, B).astype(np.float32)
  Fz = np.random.rand(M/B, N/B, B).astype(np.float32)
  test_block_hadamard(Wz, Fz=None)
  test_block_hadamard(Wz, Fz=Fz)
  test_block_Hmul(X, Wz, B, Fz=None)
  test_block_Hmul(X, Wz, B, Fz=Fz)

  # hadamard idx tests
  test_hadamard_idx(2*4, 3*4, 4)

  print("\n** ALL TESTS PASSED **")
