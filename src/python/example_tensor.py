import sys, os
import numpy as np
import singa

sys.path.append(os.path.join(os.path.dirname(__file__), '../../include/singa/proto'))
from core_pb2 import *

def tuple_to_vector(tshape):
  if type(tshape) == int:
    tshape = (tshape, 1)
  vs = singa.Shape(len(tshape))
  for i in range(len(tshape)):
    vs[i] = tshape[i]
  return vs

''' class and member functions for singa::Tensor
'''
class Tensor(object):

  def __init__(self, shape=None, device=None, dtype=kFloat32):

    self.tuple_shape = shape
    self.device = device
    self.dtype = dtype

    if shape == None:
      self.singa_tensor = singa.Tensor()
    else:
      vs = tuple_to_vector(shape)
      if device == None:
        self.singa_tensor = singa.Tensor(vs, dtype)
      else:
        self.singa_tensor = singa.Tensor(vs, device, dtype)

  def copy_Tensor(self, t):
    self.tuple_shape = t.tuple_shape
    self.device = t.device
    self.dtype = t.dtype
    self.singa_tensor = singa.Tensor(t.singa_tensor)

  def data(self):
    idx = self.singa_tensor.data_type()
    if idx == kFloat32:
      data_array = singa.floatArray_frompointer(self.singa_tensor.floatData())
      dt = np.float32
    elif idx == kFloat16:
      print 'not implemented yet'
      return
      #data_array = singa.floatArray_frompointer(self.singa_tensor.floatData())
      #dt = np.float16
    elif idx == kInt:
      data_array = singa.intArray_frompointer(self.singa_tensor.intData())
      dt = np.int32
    elif idx == kChar:
      data_array = singa.charArray_frompointer(self.singa_tensor.charData())
      dt = np.int8
    elif idx == kDouble:
      data_array = singa.doubleArray_frompointer(self.singa_tensor.doubleData())
      dt = np.float64

    data = [data_array[i] for i in range(self.singa_tensor.Size())]
    data = np.array(data, dtype=dt).reshape(self.tuple_shape)
    return data

  def data_type(self):
    return self.singa_tensor.data_type()

  def shape(self, axis=None):
    if axis==None:
       return self.singa_tensor.shape()
    else:
       return self.singa_tensor.shape(axis)

  def ndim(self):
    return self.singa_tensor.nDim()

  def transpose(self):
    return self.singa_tensor.transpose()

  def size(self):
    return self.singa_tensor.Size()

  def memsize(self):
    return self.singa_tensor.MemSize()

  def reshape(self, shape):
    self.tuple_shape = shape
    self.singa_tensor.Reshape(tuple_to_vector(shape))

  def reset_like(self, t):
    self.singa_tensor.ResetLike(t.singa_tensor)

  def as_type(self, dtype):
    self.singa_tensor.AsType(dtype)

  def to_device(self, device):
    self.singa_tensor.ToDevice(device)

  def to_host(self):
    self.singa_tensor.ToHost()


  def set_value(self, x):
    self.singa_tensor.SetValue(x)

  def copy_data_from_hostptr(self, ptr, num):
    self.singa_tensor.CopyDataFromHostPtr(ptr, num)

  def copy_data(self, t):
    self.singa_tensor.CopyData(t.singa_tensor)

  def clone(self):
    new_t = Tensor(self.tuple_shape, self.device, self.dtype)
    new_t.singa_tensor = self.singa_tensor.Clone()
    return new_t

  def matrix_transpose(self):
    new_t = Tensor(self.tuple_shape, self.device, self.dtype)
    new_t.singa_tensor = self.singa_tensor.T()
    return new_t

  # TODO(chonho-02) assign
  # same as copy_Tensor(t) ???
  def assign(self, t):
    self.singa_tensor = self.singa_tensor.Assign_T(t.singa_tensor)

  def __iadd__(self, x):
    if type(x) == Tensor:
      self.singa_tensor += x.singa_tensor
    else:
      self.singa_tensor += x

  def __isub__(self, x):
    if type(x) == Tensor:
      self.singa_tensor -= x.singa_tensor
    else:
      self.singa_tensor -= x

  def __imul__(self, x):
    if type(x) == Tensor:
      self.singa_tensor *= x.singa_tensor
    else:
      self.singa_tensor *= x

  def __idiv__(self, x):
    if type(x) == Tensor:
      self.singa_tensor /= x.singa_tensor
    else:
      self.singa_tensor /= x

  #(TODO) need a method to convert Tensor to numpy array
  #(TODO) need a method to convert numpy array to Tensor

  def copy_args(self):
    self.tuple_shape = self.singa_tensor.shape()
    self.device = self.singa_tensor.device()
    self.dtype = self.singa_tensor.data_type()

''' python functions for global methods in Tensor.h
'''
def product(shape):
  return singa.Product(tuple_to_vector(shape))

def sizeof(dtype):
  return singa.SizeOf(dtype)

def call_singa_func(_singa_func, t, *args):
  new_t = Tensor()
  new_t.singa_tensor = _singa_func(t.singa_tensor, *args)
  new_t.tuple_shape = new_t.singa_tensor.shape()
  new_t.device = new_t.singa_tensor.device()
  new_t.dtype = new_t.singa_tensor.data_type()
  return new_t

def abs(t):
  return call_singa_func(singa.Abs, t)

def exp(t):
  return call_singa_func(singa.Exp, t)

def log(t):
  return call_singa_func(singa.Log, t)

def relu(t):
  return call_singa_func(singa.ReLU, t)

def sigmoid(t):
  return call_singa_func(singa.Sigmoid, t)

def square(t):
  return call_singa_func(singa.Square, t)

def tanh(t):
  return call_singa_func(singa.Tanh, t)

def sum(t, axis=None):
  if axis == None:
    return singa.floatSum(t.singa_tensor)
  else:
    return call_singa_func(singa.Sum, t, axis)

def average(t, axis=0):
  return call_singa_func(singa.Average, t, axis)

def softmax(t, axis=0):
  return call_singa_func(singa.SoftMax, t, axis)


'''
TODO(chonho-06) remaining operators

def __lt__(t1, t2):
  return singa.Add_T(t1.singa_tensor, t2.singa_tensor)

def __add__(t1, t2):
  return singa.Add_T(t1.singa_tensor, t2.singa_tensor)

def __sub__(t1, t2):
  return singa.Sub_T(t1.singa_tensor, t2.singa_tensor)

def __mul__(t1, t2):
  return singa.Mul_T(t1.singa_tensor, t2.singa_tensor)

def __div__(t1, t2):
  return singa.Div_T(t1.singa_tensor, t2.singa_tensor)
'''


#---------------------------------------------------------
# example usage
print 'global SizeOf kFloat32:', sizeof(kFloat32)
print 'global SizeOf kFloat16:', sizeof(kFloat16)
print 'global SizeOf kInt:', sizeof(kInt)
print 'global SizeOf kDouble:', sizeof(kDouble)
print

shape = (3)
t = Tensor(shape)
print 'global Product:', product(shape)
#t.singa_tensor.AsType(kInt)
print 'shape():', t.shape()
print 'shape(0), shape(1):', t.shape(0), t.shape(1)
print 'data_type():', t.data_type()
print 'transpose', t.transpose()
print 'nDim:', t.ndim()
print 'size:', t.size()
print 'memsize:', t.memsize()
print 'data():', t.data()
print

shape = (2, 3)
t.reshape(shape)
print 'after reshape, t.shape():', t.shape()
print

print '----------------------------'
t1 = Tensor()
print 'new tensor, t1.shape():', t1.shape()
t1.copy_Tensor(t)
print 'copy t to t1, t1.shape():', t1.shape()
print 'copy t to t1, t1.data(): \n', t1.data()
print

t1.__iadd__(1.2345)
print 't1.__iadd__(1.2345), t1.data(): \n', t1.data()
print

t2 = log(t1)
print 't2 = log(t1), t2.data(): \n', t2.data()
print

t1.__iadd__(t2)
print 't1.__iadd__(t2), t1.data(): \n', t1.data()
print

t1.__imul__(2)
print 't1.__imul__(2), t1.data(): \n', t1.data()
print

print '----------------------------'
tc = t2.clone()
print 'tc = t2.clone(), tc.data(): \n', tc.data()
print

print 'sum(tc):', sum(tc)
t3 = sum(tc,0)
print 't3 = sum(tc,0) \n', t3.data()
t3 = sum(tc,1)
print 't3 = sum(tc,1) \n', t3.data()

t3 = average(tc,0)
print 't3 = average(tc,0) \n', t3.data()
t3 = average(tc,1)
print 't3 = average(tc,0) \n', t3.data()

t3 = softmax(tc)
print t3.data()

print '----------------------------'


#devCPU = singa.CppCPU(1)
#devGPU = singa.CudaGPU(2)
