import sys, os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),
                             '../../build/python'))
sys.path.append(os.path.join(os.path.dirname(__file__),
                             '../../build/lib'))
sys.path.append(os.path.join(os.path.dirname(__file__),'../../build/src'))
from core_pb2 import *
from singa.tensor import *


#---------------------------------------------------------
# example usage
#---------------------------------------------------------

print '----------------------------'
print 'global SizeOf kFloat32:', sizeof(kFloat32)
print 'global SizeOf kFloat16:', sizeof(kFloat16)
print 'global SizeOf kInt:', sizeof(kInt)
print 'global SizeOf kDouble:', sizeof(kDouble)
print

a = Tensor()
print 'a = Tensor()'
print 'only defaultdevice is assigned \n'

shape = (1, 6)
t = Tensor(shape)
print 'shape = (1, 6):', t.shape()
print 'shape(0), shape(1):', t.shape(0), t.shape(1)
print 'global Product:', product(shape)
print 't = Tensor(shape)'
#t.singa_tensor.AsType(kInt)
print 'data_type():', t.data_type()
print 'transpose', t.is_transpose()
print 'nDim:', t.ndim()
print 'size:', t.size()
print 'memsize:', t.memsize()
print 'data():', t.to_numpy()
print

print '----------------------------'
print 't.data\n', t.to_numpy()
q = t.copy()
w = t.deepcopy()
print 'q.data\n', q.to_numpy()
print 'w.data\n', w.to_numpy()
t += 1.23
print 'q.data\n', q.to_numpy()
print 'w.data\n', w.to_numpy()

print '----------------------------'
shape = (2, 3)
t.reshape(shape)
print 'shape = (3, 2)'
print 'after reshape, t.shape():', t.shape()
print 't.data(): \n', t.to_numpy()
shape = (3, 2)
t0 = reshape(t, shape)
print 'shape = (2, 3)'
print 'after t0 = reshape(t, shape) \n'
print 't.shape():', t.shape()
print 't0.shape():', t0.shape()
print

print '----------------------------'
t += 1.2345
print 't += 1.234, i.e., t.__iadd__(1.2345): \n', t.to_numpy()
print

t1 = t
print 'copy\nt1 = t'
print 't1.shape():', t1.shape()
print 't1.data(): \n', t1.to_numpy()
print

r = t1.transpose()
print 'r = t1.transpose()\nr.data() \n', r.to_numpy()


print '----------------------------'
t2 = log(t1)
print 't2 = log(t1): \n', t2.to_numpy()
print

t1 += t2
print 't1 += t2, i.e., t1.__iadd__(t2): \n', t1.to_numpy()
print

t1 *= 2
print 't1 *= 2, i.e., t1.__imul__(2): \n', t1.to_numpy()
print

print '----------------------------'
tc = t2.clone()
print 'clone\ntc = t2.clone()\ntc.data(): \n', tc.to_numpy()
print

print 'sum(tc) \n', sum(tc)
print
t3 = sum(tc,0)
print 'sum(tc,0) \n', t3.to_numpy()
t3 = sum(tc,1)
print 'sum(tc,1) \n', t3.to_numpy()
print

t3 = average(tc,0)
print 'average(tc,0) \n', t3.to_numpy()
t3 = average(tc,1)
print 'average(tc,1) \n', t3.to_numpy()
print

t3 = softmax(tc,0)
print 'softmax(tc,0)\n', t3.to_numpy()
t3 = softmax(tc,1)
print 'softmax(tc,1)\n', t3.to_numpy()

print '----------------------------'
print 't1 \n', t1.to_numpy()
print

n = t1 + t2
print 't1 + t2: \n', n.to_numpy()
print

n = t1 * t2
print 't1*t2: \n', n.to_numpy()
print

n = t1 - 1.2
print 't1 - 1.2 \n', n.to_numpy()
print

n = add(t1, t1)
print 'add(t1, t1) \n', n.to_numpy()
print

n = add(t1, 3.4)
print 'add(t1, 3.4) \n', n.to_numpy()
print

n = div(t1, 2.0)
print 'div(t1, 2.0) \n', n.to_numpy()
print

print '----------------------------'
shape = (2, 2)
t4 = Tensor(shape)
t4 += 3.45
print 't4 += 3.45 \n', t4.to_numpy()
print

n = t4 < 3.45
print 't4 < 3.45 \n', n.to_numpy()
print

n = lt(t4, 3.45)
print 'lt(t4, 3.45) \n', n.to_numpy()
print

n = ge(t4, 3.45)
print 'ge(t4, 3.45) \n', n.to_numpy()
print

print '----------------------------'
print t1.to_numpy()
print tc.to_numpy()
print
copy_data_to_from(t1, tc, 2)
print t1.to_numpy()
print tc.to_numpy()

#ttt = t1.singa_tensor < 5.2
#ttt = lessthan(t1, 5.2)
#print ttt.data()

#devCPU = singa.CppCPU(1)
#devGPU = singa.CudaGPU(2)
