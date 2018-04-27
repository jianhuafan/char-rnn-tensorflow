import numpy as np
import scipy.linalg

hadamard_dict = {}
haar_dict = {}

def hartley(N, dtype=np.float32):
  F = scipy.linalg.dft(N, scale='sqrtn')
  return np.real(F) - np.imag(F)

def hadamard(N, dtype=np.float32):
  """ Returns N-by-N normalized Hadamard matrix, use memoization """
  if N in hadamard_dict:
    return hadamard_dict[N]
  elif N == 2:
    H = np.array([[1.,1.],[1.,-1.]], dtype=dtype) /  np.sqrt(2)
    hadamard_dict[N] = H
    return H
  else:
    assert(N%2 == 0)
    H_ = hadamard(N/2, dtype=dtype)
    H = np.empty((N,N), dtype=dtype)
    H[0:N/2,0:N/2] = H_
    H[0:N/2,N/2:N] = H_
    H[N/2:N,0:N/2] = H_
    H[N/2:N,N/2:N] = -H_
    H = H / np.sqrt(2)
    hadamard_dict[N] = H
    return H

def walsh(N, dtype=np.float32):
  """ Returns N-by-N normalized Walsh matrix, which is the
      Hadamard matrix in sequency order """
  def reverse_bits(x, bits=8):
    y = x & 1
    for i in range(1,bits):
      y = y << 1
      y = y | ((x >> i) & 1)
    return y

  H = hadamard(N, dtype=dtype)
  # bit reversal permutation
  bits = np.int_(np.round(np.log2(N)))
  b = [reverse_bits(x, bits=bits) for x in np.arange(N)]
  # gray code permutation
  g = [x ^ (x >> 1) for x in np.arange(N)]
  H = H[b]
  H = H[g]
  return H

def haar(N, dtype=np.float32):
  """ Returns N-by-N normalized Haar matrix """
  if N in haar_dict:
    return haar_dict[N]
  elif N == 2:
    H = np.array([[1.,1.],[1.,-1.]], dtype=dtype) /  np.sqrt(2)
    haar_dict[N] = H
    return H
  else:
    assert(N%2 == 0)
    H_ = haar(N/2, dtype=dtype)
    H = np.empty((N,N), dtype=dtype)
    H[0:N/2,:] = np.kron(H_, np.array([1.,1.]))
    H = H / np.sqrt(2)
    H[N/2:N,:] = np.kron(np.eye(N/2), np.array([1.,-1.])/np.sqrt(2.))
    haar_dict[N] = H
    return H

def fht_rec(x,y):
  """ Fast Hadamard transform of vector x, uses n log n adds """
  N = x.shape[-1]

  y[..., 0:N:2] = x[..., 0:N:2] + x[..., 1:N:2]
  y[..., 1:N:2] = x[..., 0:N:2] - x[..., 1:N:2]

  n = 2
  while n < N:
    k = n
    n = n * 2
    for i in range(0,N,n):
      yt              = y[..., i:i+k] + y[..., i+k:i+n]
      y[..., i+k:i+n] = y[..., i:i+k] - y[..., i+k:i+n]
      y[..., i  :i+k] = yt

  return y/np.sqrt(N)

def fht(x):
  xshape = x.shape
  y = np.empty_like(x)
  y = fht_rec(x,y)
  #_libh.fht(y, x, x.shape[0], x.size//x.shape[0])
  return y

def ifht(x):
  return fht(x)
  #xshape = x.shape
  #y = np.empty_like(x)
  #y = fht_rec(x,y) / x.shape[-1]
  ##_libh.ifht(y, x, x.shape[0], x.size//x.shape[0])
  #return y

#-------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------
if __name__ == "__main__":
  np.set_printoptions(precision=3)
  N = 8
  A = np.random.rand(N,N).astype(np.float32)

  F = scipy.linalg.dft(N,scale='sqrtn')
  H = hartley(N)

  # Hartley is involutory
  B = np.matmul(H, np.matmul(H,A))
  assert(np.allclose(A,B))

  # check Hartley
  FA = np.matmul(F,A)
  C1 = np.matmul(H,A)
  C2 = np.real(FA) - np.imag(FA)
  assert(np.allclose(C1,C2))

  print("\n** ALL TESTS PASSED **")
