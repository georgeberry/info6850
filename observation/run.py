'''
give groups of 4 arguments:
durationA, durationB, summary_output.csv, full_output.csv

arg[0] is for system
arg[1] and arg[2] should be the file paths of the raw data
after that, groups of 4
'''

import sys
from fsqr import *
import gc

if __name__ == '__main__':
    filepaths = sys.argv[1:3]
    args = sys.argv[3:]
    args.reverse() #reverse and then pop from end
    print filepaths, args
    if len(args) % 4 != 0:
        print 'wrong number of arguments, must be multiple of 4'
    else:
        for iteration in xrange(len(args)/4):
            f = fsqr(filepaths[0], filepaths[1])
            f(int(args.pop()), int(args.pop()), args.pop(), args.pop())
            del f
            gc.collect()