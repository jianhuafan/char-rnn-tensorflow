import sys, os, re
import subprocess as sp
import timeit

block_size = [64, 2, 4, 8, 16, 32]

if __name__ == "__main__":
  for B in block_size:

    sp.call(["python", "train.py",
             "--block_size=%d" % B]
             )
    sp.call(["python", "train.py",
             "--block_size=%d" % B,
             "--num_epochs=%d" % 1,
             "--train_flag=False",
             "--init_from=save_{}".format(B)]
             )
