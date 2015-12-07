#!/usr/bin/env python


import os
import sys
import string
import pb2.job_pb2 as job_pb2
import singa.driver as driver
from google.protobuf.text_format import Merge

if __name__ == '__main__':
    i =  sys.argv.index("-conf")
    s = open(sys.argv[i+1], 'r').read()
    s = str(s)
    j = job_pb2.JobProto()  
    Merge(s,j)
    b = j.SerializeToString()
    d = driver.Driver()
    d.Init(sys.argv)
    d.Train(False,b)
