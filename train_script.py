import sys, os, re
import subprocess as sp
import timeit

block_size = [2, 4, 8]
transform_list = ["Hadamard", "Fourier"]
w_bit_list = [2, 3, 4, 8, 16]
quant = "bit"
model = 'lstm'

if __name__ == "__main__":
  for w_bit in w_bit_list:
    f_bit = w_bit
    sp.call(["python", "train.py",
              "--quant=%s" % quant,
              "--model=%s" % model,
              "--w_bit=%d" % w_bit,
              "--f_bit=%d" % f_bit]
             )
    sp.call(["python", "train.py",
             "--quant=%s" % quant,
             "--model=%s" % model,
             "--w_bit=%d" % w_bit,
             "--f_bit=%d" % f_bit,
             "--num_epochs=%d" % 1,
             "--test_flag=True",
             "--init_from=save_bit_{}".format(w_bit)]
             )
