import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__),
                             '../../src/python'))
from tensor import *
from device import *

sys.path.append(os.path.join(os.path.dirname(__file__),
                             '../../src'))
from core_pb2 import *

#---------------------------------------------------------
# example usage
#---------------------------------------------------------

d1 = CudaGPU(123)
print d1.singa_device
print d1.get_host()
print d1.get_id()
print

d2 = CppCPU(345)
print d2.singa_device
print d2.get_host()
print d2.get_id()
print

s = (2, 3)
t = Tensor(s, d2.get_host())
print t.singa_tensor
print t.device
print

d = Device(0)
print d.singa_device
print
