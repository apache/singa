import sys, os
import numpy as np
import singa

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from core_pb2 import *


''' Class and member functions for singa::Tensor
'''
class Tensor(object):

  def __init__(self, shape=None, device=None, dtype=kFloat32):
    ''' shape = (tuple)
    '''
    if shape == None:
      # call constructor of singa::Tensor
      self.singa_tensor = singa.Tensor()
      return
    else:
      assert type(shape) == tuple, 'shape should be tuple'
      vs = _tuple_to_vector(shape)
      if device == None:
        self.singa_tensor = singa.Tensor(vs, dtype)
      else:
        self.singa_tensor = singa.Tensor(vs, device, dtype)
      self.tuple_shape = shape
      self.device = device
      self.dtype = dtype

  def toarray(self):
    #TODO(chonho) - need to think more efficient way to convert???
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

  def is_transpose(self):
    return self.singa_tensor.transpose()

  def size(self):
    return self.singa_tensor.Size()

  def memsize(self):
    return self.singa_tensor.MemSize()

  def reshape(self, shape):
    assert product(self.tuple_shape) == product(shape), \
           'product of shape should be equal'
    self.tuple_shape = shape
    self.singa_tensor.Reshape(_tuple_to_vector(shape))

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

  def copy_data(self, t):
    self.singa_tensor.CopyData(t.singa_tensor)

  def clone(self):
    ''' it does deep copy
        call singa::Tensor::Clone()
    '''
    return _call_singa_func(self.singa_tensor.Clone)

  def transpose(self):
    ''' shallow copy, negate the transpose field
        call singa::Tensor::T()
    '''
    return _call_singa_func(self.singa_tensor.T)

  def copy(self):
    ''' shallow copy
        call copy constructor of singa::Tensor
    '''
    return _call_singa_func(singa.Tensor, self.singa_tensor)

  def deepcopy(self):
    ''' deep copy
        call singa::Tensor::Clone()
    '''
    return self.clone()

  '''
  python operators (+=, -=, *=, /=) for singa::Tensor unary operators
  '''
  def __iadd__(self, x):
    if type(x) == Tensor:
      self.singa_tensor += x.singa_tensor
    else:
      self.singa_tensor += x
    return self

  def __isub__(self, x):
    if type(x) == Tensor:
      self.singa_tensor -= x.singa_tensor
    else:
      self.singa_tensor -= x
    return self

  def __imul__(self, x):
    if type(x) == Tensor:
      self.singa_tensor *= x.singa_tensor
    else:
      self.singa_tensor *= x
    return self

  def __idiv__(self, x):
    if type(x) == Tensor:
      self.singa_tensor /= x.singa_tensor
    else:
      self.singa_tensor /= x
    return self

  '''
  python operators (+, -, *, /, <, <=, >, >=) for singa binary operators
  '''
  def __add__(self, rhs):
    if isinstance(rhs, Tensor):
      return _call_singa_func(singa.Add_TT, self.singa_tensor, rhs.singa_tensor)
    else:
      return _call_singa_func(singa.Add_Tf, self.singa_tensor, rhs)

  def __sub__(self, rhs):
    if isinstance(rhs, Tensor):
      return _call_singa_func(singa.Sub_TT, self.singa_tensor, rhs.singa_tensor)
    else:
      return _call_singa_func(singa.Sub_Tf, self.singa_tensor, rhs)

  def __mul__(self, rhs):
    if isinstance(rhs, Tensor):
      return _call_singa_func(singa.Mul_TT, self.singa_tensor, rhs.singa_tensor)
    else:
      return _call_singa_func(singa.Mul_Tf, self.singa_tensor, rhs)

  def __div__(self, rhs):
    if isinstance(rhs, Tensor):
      return _call_singa_func(singa.Div_TT, self.singa_tensor, rhs.singa_tensor)
    else:
      return _call_singa_func(singa.Div_Tf, self.singa_tensor, rhs)

  def __lt__(self, rhs):
    return _call_singa_func(singa.LT_Tf, self.singa_tensor, rhs)

  def __le__(self, rhs):
    return _call_singa_func(singa.LE_Tf, self.singa_tensor, rhs)

  def __gt__(self, rhs):
    return _call_singa_func(singa.GT_Tf, self.singa_tensor, rhs)

  def __ge__(self, rhs):
    return _call_singa_func(singa.GE_Tf, self.singa_tensor, rhs)



''' python functions for global functions in Tensor.h
'''
def product(shape):
  return reduce(lambda x, y: x * y, shape)

def sizeof(dtype):
  return singa.SizeOf(dtype)

def reshape(t, s):
  return _call_singa_func(singa.Reshape, t.singa_tensor, s)

def copy_data_to_from(dst, src, size, src_offset=0, dst_offset=0):
  singa.CopyDataToFrom(dst.singa_tensor, src.singa_tensor, size,
                       src_offset, dst_offset)

def abs(t):
  return _call_singa_func(singa.Abs, t.singa_tensor)

def exp(t):
  return _call_singa_func(singa.Exp, t.singa_tensor)

def log(t):
  return _call_singa_func(singa.Log, t.singa_tensor)

def relu(t):
  return _call_singa_func(singa.ReLU, t.singa_tensor)

def sigmoid(t):
  return _call_singa_func(singa.Sigmoid, t.singa_tensor)

def square(t):
  return _call_singa_func(singa.Square, t.singa_tensor)

def tanh(t):
  return _call_singa_func(singa.Tanh, t.singa_tensor)

def sum(t, axis=None):
  if axis == None:
    return singa.floatSum(t.singa_tensor)
  else:
    return _call_singa_func(singa.Sum, t.singa_tensor, axis)

def pow(t, x):
  print 'not implemented yet'

def average(t, axis=0):
  return _call_singa_func(singa.Average, t.singa_tensor, axis)

def softmax(t, axis=0):
  return _call_singa_func(singa.SoftMax, t.singa_tensor, axis)

def lt(t, x):
  return t < x

def le(t, x):
  return t <= x

def gt(t, x):
  return t > x

def ge(t, x):
  return t >= x

def add(lhs, rhs):
  # call Tensor.__add__()
  return lhs + rhs

def sub(lhs, rhs):
  # call Tensor.__sub__()
  return lhs - rhs

def eltwise_mult(lhs, rhs):
  # call Tensor.__mul__()
  return lhs * rhs

def div(lhs, rhs):
  # call Tensor.__div__()
  return lhs / rhs


''' private functions, internally used
'''
def _tuple_to_vector(tshape):
  ''' this function converts tuple to std::vector<int>
  '''
  vs = singa.Shape(len(tshape))
  for i in range(len(tshape)):
    vs[i] = tshape[i]
  return vs

def _call_singa_func(_singa_func, *args):
  ''' this function calls singa global functions that returns Tensor
      and create new python Tensor instance
      e.g., Tensor [singa_func](args...)
  '''
  new_t = Tensor()
  new_t.singa_tensor = _singa_func(*args)
  new_t.tuple_shape = new_t.singa_tensor.shape()
  new_t.device = new_t.singa_tensor.device()
  new_t.dtype = new_t.singa_tensor.data_type()
  return new_t
