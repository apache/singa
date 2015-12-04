import os
import sys
import string
import driver
from optparse import OptionParser

if __name__ == '__main__':
    d = driver.Driver();
    d.Init(sys.argv);
    i =  sys.argv.index("-conf")
    s = open(sys.argv[i+1], 'r').read()
    s = str(s)
    print s
    d.Train(False, s)

