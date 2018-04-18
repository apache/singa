# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# =============================================================================
"""
Example usage::

    import numpy as np
    from singa import tensor
    from singa import device

    # create a tensor with shape (2,3), default CppCPU device and float32
    x = tensor.Tensor((2, 3))
    x.set_value(0.4)

    # create a tensor from a numpy array
    npy = np.zeros((3, 3), dtype=np.float32)
    y = tensor.from_numpy(npy)

    y.uniform(-1, 1)  # sample values from the uniform distribution

    z = tensor.mult(x, y)  # gemm -> z of shape (2, 3)

    x += z  # element-wise addition

    dev = device.get_default_device()
    x.to_device(dev)  # move the data to a gpu device

    r = tensor.relu(x)

    s = tensor.to_numpy(r)  # tensor -> numpy array

There are two sets of tensor functions,

Tensor member functions
    which would change the internal state of the Tensor instance.

Tensor module functions
    which accept Tensor instances as arguments and return Tensor instances.

Every Tesor instance must be initialized before reading data from it.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import object
import numpy as np
from functools import reduce

from .proto import core_pb2
from . import singa_wrap as singa
from . import device as pydevice

int32 = core_pb2.kInt
float32 = core_pb2.kFloat32


class Tensor(object):
    '''Create a Py Tensor, which wraps a swig converted Tensor from CPP Tensor

    The three arguments are three attributes of the Tensor.

    Args:
        shape (list<int>): a list of integers for the tensor shape. If shape is
            not specified, the created tensor is called a dummy tensor.
        device: a swig converted Device instance using the device moduel . If it
            is None, then the default host device would be used.
        dtype: data type. currently, most operations only accept kFloat32.
    '''

    def __init__(self, shape=None, device=None, dtype=core_pb2.kFloat32):
        if shape is None:
            # call constructor of singa::Tensor
            self.singa_tensor = singa.Tensor()
            return
        else:
            assert isinstance(shape, tuple), 'shape should be tuple'
            if device is None:
                device = pydevice.get_default_device()
                self.singa_tensor = singa.Tensor(list(shape), device, dtype)
            else:
                self.singa_tensor = singa.Tensor(list(shape), device, dtype)
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def ndim(self):
        '''
        Returns:
            the number of dimensions of the tensor.
        '''
        return self.singa_tensor.nDim()

    def is_empty(self):
        '''
        Returns:
            True if the tensor is empty according to its shape
        '''
        return self.ndim() == 0

    def is_transpose(self):
        '''
        Returns:
            True if the internal data is transposed; otherwise False.
        '''
        return self.singa_tensor.transpose()

    def size(self):  # TODO(wangwei) compute size
        '''
        Returns:
            the number of elements of the tensor.
        '''
        return self.singa_tensor.Size()

    def memsize(self):
        '''
        Returns:
            the number of Bytes allocated for this tensor.
        '''
        return self.singa_tensor.MemSize()

    def reshape(self, shape):
        '''Change the tensor shape.

        Args:
            shape (list<int>): new shape, which should have the same volumn as
                the original shape.
        '''
        assert product(self.shape) == product(shape), \
            'product of shape should be equal'
        self.shape = shape
        self.singa_tensor.Reshape(list(shape))

    def reset_like(self, t):
        '''Reset the shape, dtype and device as the given tensor.

        Args:
            t (Tensor)
        '''
        self.singa_tensor.ResetLike(t.singa_tensor)
        self.shape = t.shape
        self.device = t.device
        self.dtype = t.dtype

    '''
    def as_type(self, dtype):
        Change the data type.

        Args:
            dtype:
        self.singa_tensor.AsType(dtype)
    '''

    def to_device(self, device):
        '''Move the tensor data onto a given device.

        Args:
            device: a swig Device converted from CudaGPU or CppCPU or OpenclGPU
        '''
        self.singa_tensor.ToDevice(device)
        self.device = device

    def to_host(self):
        '''Move the tensor data onto the default host CppCPU device.
        '''
        self.singa_tensor.ToHost()
        self.device = pydevice.default_device

    def l2(self):
        '''
        Returns:
            the L2 norm.
        '''
        return self.singa_tensor.L2()

    def l1(self):
        '''
        Returns:
            the L1 norm.
        '''
        return self.singa_tensor.L1()

    def set_value(self, x):
        '''Set all elements of the tensor to be the give value.

        Args:
            x (float), a float value to be set to all elements.
        '''
        # assert type(x) == float, 'set value only accepts float input'
        # if isinstance(x, float):
        self.singa_tensor.SetFloatValue(float(x))

    def copy_from_numpy(self, np_array, offset=0):
        ''' Copy the data from the numpy array.

        Args:
            np_array: source numpy array
            offset (int): destination offset
        '''
        assert np_array.size == self.size(), 'tensor shape should be the same'
        if not np_array.ndim == 1:
            np_array = np_array.flatten()
        dt = np_array.dtype
        if dt == np.float32:
            self.singa_tensor.CopyFloatDataFromHostPtr(np_array)
        elif dt == np.int or dt == np.int32:
            self.singa_tensor.CopyIntDataFromHostPtr(np_array)
        else:
            print('Not implemented yet for ', dt)

    def copy_data(self, t):
        '''Copy data from other Tensor instance.

        Args:
            t (Tensor): source Tensor.
        '''
        assert isinstance(t, Tensor), 't must be a singa Tensor instance'
        self.singa_tensor.CopyData(t.singa_tensor)

    def clone(self):
        '''
        Returns:
            a new Tensor which does deep copy of this tensor
        '''
        return _call_singa_func(self.singa_tensor.Clone)

    def T(self):
        ''' shallow copy, negate the transpose field.

        Returns:
            a new Tensor which shares the underlying data memory (shallow copy)
            but is marked as a transposed version of this tensor.
        '''
        return _call_singa_func(self.singa_tensor.T)

    def copy(self):
        '''shallow copy calls copy constructor of singa::Tensor
        '''
        return _call_singa_func(singa.Tensor, self.singa_tensor)

    def deepcopy(self):
        '''Same as clone().

        Returns:
            a new Tensor
        '''
        return self.clone()

    def bernoulli(self, p):
        '''Sample 0/1 for each element according to the given probability.

        Args:
            p (float): with probability p, each element is sample to 1.
        '''
        singa.Bernoulli(float(p), self.singa_tensor)

    def gaussian(self, mean, std):
        '''Generate a value for each element following a Gaussian distribution.

        Args:
            mean (float): mean of the distribution
            std (float): standard variance of the distribution
        '''
        singa.Gaussian(float(mean), float(std), self.singa_tensor)

    def uniform(self, low, high):
        '''Generate a value for each element following a uniform distribution.

        Args:
            low (float): the lower bound
            high (float): the hight bound
        '''
        singa.Uniform(float(low), float(high), self.singa_tensor)

    def add_column(self, v):
        '''Add a tensor to each column of this tensor.

        Args:
            v (Tensor): a Tensor to be added as a column to this tensor.
        '''
        singa.AddColumn(v.singa_tensor, self.singa_tensor)

    def add_row(self, v):
        '''Add a tensor to each row of this tensor.

        Args:
            v (Tensor): a Tensor to be added as a row to this tensor.
        '''
        singa.AddRow(v.singa_tensor, self.singa_tensor)

    def div_column(self, v):
        '''Divide each column of this tensor by v.

        Args:
            v (Tensor): 1d tensor of the same length the column of self.
        '''
        singa.DivColumn(v.singa_tensor, self.singa_tensor)

    def div_row(self, v):
        '''Divide each row of this tensor by v.

        Args:
            v (Tensor): 1d tensor of the same length the row of self.
        '''
        singa.DivRow(v.singa_tensor, self.singa_tensor)

    def mult_column(self, v):
        '''Multiply each column of this tensor by v element-wisely.

        Args:
            v (Tensor): 1d tensor of the same length the column of self.
        '''
        singa.MultColumn(v.singa_tensor, self.singa_tensor)

    def mult_row(self, v):
        '''Multiply each row of this tensor by v element-wisely.

        Args:
            v (Tensor): 1d tensor of the same length the row of self.
        '''
        singa.MultRow(v.singa_tensor, self.singa_tensor)

    '''
    python operators (+=, -=, *=, /=) for singa::Tensor unary operators
    '''

    def __iadd__(self, x):
        ''' inplace element-wise addition with a tensor or a float value.

        Args:
            x (float or Tensor):
        '''
        if isinstance(x, Tensor):
            self.singa_tensor += x.singa_tensor
        else:
            self.singa_tensor += float(x)
        return self

    def __isub__(self, x):
        ''' inplace element-wise subtraction with a tensor or a float value.

        Args:
            x (float or Tensor):
        '''

        if isinstance(x, Tensor):
            self.singa_tensor -= x.singa_tensor
        else:
            self.singa_tensor -= float(x)
        return self

    def __imul__(self, x):
        ''' inplace element-wise multiplication with a tensor or a float value.

        Args:
            x (float or Tensor):
        '''
        if isinstance(x, Tensor):
            self.singa_tensor *= x.singa_tensor
        else:
            self.singa_tensor *= float(x)
        return self

    def __idiv__(self, x):
        ''' inplace element-wise division by a tensor or a float value.

        Args:
            x (float or Tensor):
        '''
        if isinstance(x, Tensor):
            self.singa_tensor /= x.singa_tensor
        else:
            self.__imul__(1/float(x))
        return self

    '''
    python operators (+, -, *, /, <, <=, >, >=) for singa binary operators
    https://docs.python.org/2/library/operator.html#mapping-operators-to-functions
    '''

    def __add__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__add__(self.singa_tensor, rhs.singa_tensor))
        else:
            return _call_singa_func(singa.AddFloat,
                                    self.singa_tensor, rhs)

    def __sub__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__sub__(self.singa_tensor, rhs.singa_tensor))
        else:
            return _call_singa_func(singa.SubFloat,
                                    self.singa_tensor, rhs)

    def __mul__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__mul__(self.singa_tensor, rhs.singa_tensor))
        else:
            return _call_singa_func(singa.MultFloat,
                                    self.singa_tensor, rhs)

    def __div__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__div__(self.singa_tensor, rhs.singa_tensor))
        else:
            return _call_singa_func(singa.DivFloat,
                                    self.singa_tensor, rhs)

    def __truediv__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__div__(self.singa_tensor, rhs.singa_tensor))
        else:
            return _call_singa_func(singa.DivFloat,
                                    self.singa_tensor, rhs)

    def __lt__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__lt__(self.singa_tensor, rhs.singa_tensor))
        else:
            return _call_singa_func(singa.LTFloat, self.singa_tensor, rhs)

    def __le__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__le__(self.singa_tensor, rhs.singa_tensor))
        else:
            return _call_singa_func(singa.LEFloat, self.singa_tensor, rhs)

    def __gt__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__gt__(self.singa_tensor, rhs.singa_tensor))
        else:
            return _call_singa_func(singa.GTFloat, self.singa_tensor, rhs)

    def __ge__(self, rhs):
        if isinstance(rhs, Tensor):
            return from_raw_tensor(
                singa.__ge__(self.singa_tensor, rhs.singa_tensor))
        else:
            return _call_singa_func(singa.GEFloat, self.singa_tensor, rhs)

    def __radd__(self, lhs):
        lhs = float(lhs)
        one = Tensor(self.shape, self.device, self.dtype)
        one.set_value(lhs)
        one += self
        return one

    def __rsub__(self, lhs):
        lhs = float(lhs)
        one = Tensor(self.shape, self.device, self.dtype)
        one.set_value(lhs)
        one -= self
        return one

    def __rmul__(self, lhs):
        lhs = float(lhs)
        one = Tensor(self.shape, self.device, self.dtype)
        one.set_value(lhs)
        one *= self
        return one

    def __rdiv__(self, lhs):
        lhs = float(lhs)
        one = Tensor(self.shape, self.device, self.dtype)
        one.set_value(lhs)
        one /= self
        return one

    def __rtruediv__(self, lhs):
        lhs = float(lhs)
        one = Tensor(self.shape, self.device, self.dtype)
        one.set_value(lhs)
        one /= self
        return one

''' python functions for global functions in Tensor.h
'''


def from_raw_tensor(t):
    x = Tensor(t.shape(), t.device(), t.data_type())
    x.singa_tensor = t
    return x


def from_raw_tensors(tt):
    ret = []
    for t in list(tt):
        ret.append(from_raw_tensor(t))
    return ret


def product(shape):
    return reduce(lambda x, y: x * y, shape)


def sizeof(dtype):
    '''
    Returns:
        the number of bytes of the given SINGA data type defined in core.proto
    '''
    return singa.SizeOf(dtype)


def reshape(t, s):
    '''Reshape the input tensor with the given shape.

    Args:
        t (Tensor): the tensor to be changed
        s (list<int>): the new shape, which should have the same volumn as the
            old shape.

    Returns:
        the new Tensor
    '''
    return _call_singa_func(singa.Reshape, t.singa_tensor, s)


def copy_data_to_from(dst, src, size, dst_offset=0, src_offset=0):
    '''Copy the data between two Tensor instances which could be on different
    devices.

    Args:
        dst (Tensor): destination Tensor
        src (Tensor): source Tensor
        size (int) : number of elements to copy
        dst_offset (int): offset in terms of elements to the start of dst
        src_offset (int): offset in terms of elements to the start of src
    '''
    singa.CopyDataToFrom(dst.singa_tensor, src.singa_tensor, size,
                         dst_offset, src_offset)


def from_numpy(np_array):
    '''Create a Tensor instance with the shape, dtype and values from the numpy
    array.

    Args:
        np_array: the numpy array.

    Returns:
        A Tensor instance allocated on the default CppCPU device.
    '''
    assert type(np_array) is np.ndarray, 'Must input numpy array'
    # convert to float32 array
    if np_array.dtype == np.float64 or np_array.dtype == np.float:
        np_array = np_array.astype(np.float32)

    if np_array.dtype == np.int64 or np_array.dtype == np.int:
        np_array = np_array.astype(np.int32)

    if np_array.dtype == np.float32:
        dtype = core_pb2.kFloat32
    else:
        assert np_array.dtype == np.int32, \
            'Only float and int tensors are supported'
        dtype = core_pb2.kInt
    ret = Tensor(np_array.shape, dtype=dtype)
    ret.copy_from_numpy(np_array)
    return ret


def to_host(t):
    '''Copy the data to a host tensor.
    '''
    ret = t.clone()
    ret.to_host()
    return ret


def to_numpy(t):
    '''Copy the tensor into a numpy array.

    Args:
        t (Tensor), a Tensor

    Returns:
        a numpy array
    '''
    th = to_host(t)
    if th.dtype == core_pb2.kFloat32:
        np_array = th.singa_tensor.GetFloatValue(int(th.size()))
    elif th.dtype == core_pb2.kInt:
        np_array = th.singa_tensor.GetIntValue(int(th.size()))
    else:
        print('Not implemented yet for ', th.dtype)
    return np_array.reshape(th.shape)


def abs(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = abs(x), x is an element of t
    '''
    return _call_singa_func(singa.Abs, t.singa_tensor)


def exp(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = exp(x), x is an element of t
    '''
    return _call_singa_func(singa.Exp, t.singa_tensor)


def log(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = log(x), x is an element of t
    '''
    return _call_singa_func(singa.Log, t.singa_tensor)


def relu(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = x if x >0; otherwise 0; x is an element
        of t
    '''
    return _call_singa_func(singa.ReLU, t.singa_tensor)


def sigmoid(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = sigmoid(x); x is an element of t
    '''
    return _call_singa_func(singa.Sigmoid, t.singa_tensor)


def sign(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = sign(x)
    '''
    return _call_singa_func(singa.Sign, t.singa_tensor)


def sqrt(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = sqrt(x), x is an element of t
    '''
    return _call_singa_func(singa.Sqrt, t.singa_tensor)


def square(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = x * x, x is an element of t
    '''
    return _call_singa_func(singa.Square, t.singa_tensor)


def tanh(t):
    '''
    Args:
        t (Tensor): input Tensor

    Returns:
        a new Tensor whose element y = tanh(x), x is an element of t
    '''
    return _call_singa_func(singa.Tanh, t.singa_tensor)


def sum(t, axis=None):
    '''Sum elements of the input tensor long the given axis.

    Args:
        t (Tensor): input Tensor
        axis (int, optional): if None, the summation is done over all elements;
            if axis is provided, then it is calculated along the given axis,
            e.g. 0 -- sum each column; 1 -- sum each row.

    Returns:
        a float value as the sum of all elements, or a new Tensor
    '''

    if axis is None:
        return singa.SumAsFloat(t.singa_tensor)
    else:
        return _call_singa_func(singa.Sum, t.singa_tensor, axis)


def pow(t, x, out=None):
    '''
    Args:
        t (Tensor): input tensor
        x (float or Tensor): y[i] = t[i]^x if x is a float value; otherwise,
            y[i]= t[i]^x[i] if x is a tensor.
        out (None or Tensor): if None, a new Tensor would be constructed to
            store the result; otherwise, the result is put into out.

    Returns:
        the result tensor.
    '''
    if out is None:
        if isinstance(x, Tensor):
            return _call_singa_func(singa.Pow, t.singa_tensor, x.singa_tensor)
        else:
            return _call_singa_func(singa.PowFloat, t.singa_tensor, x)
    else:
        if isinstance(x, Tensor):
            singa.PowWithRet(t.singa_tensor, x.singa_tensor, out.singa_tensor)
        else:
            singa.PowFloatWitRet(t.singa_tensor, x, out.singa_tensor)
        return out


def average(t, axis=None):
    '''
    Args:
        t (Tensor): input Tensor
        axis (int, optional): if None, average all elements; otherwise average
            along the given dimension. 0 for averaging each column; 1 for
            averaging each row.

    Returns:
        a float value if axis is None; otherwise, a new Tensor for the result.
    '''
    if t.ndim() > 1:
        return _call_singa_func(singa.Average, t.singa_tensor, axis)
    else:
        return singa.SumAsFloat(t.singa_tensor) / t.size()


def softmax(t, out=None):
    '''Apply SoftMax for each row of the Tensor.

    Args:
        t (Tensor): the input 1d or 2d tensor
        out (Tensor, optional): if not None, it is used to store the result

    Returns:
        the result Tensor
    '''
    if out is None:
        return _call_singa_func(singa.SoftMax, t.singa_tensor)
    else:
        singa.SoftMax(t.singa_tensor, out.singa_tensor)
        return out


def lt(t, x):
    '''Elementi-wise comparison for t < x

    Args:
        t (Tensor): left hand side operand
        x (Tensor or float): right hand side operand

    Returns:
        a Tensor with each element being t[i] < x ? 1.0f:0.0f,
        or t[i] < x[i] ? 1.0f:0.0f
    '''
    return t < x


def le(t, x):
    '''Elementi-wise comparison for t <= x.

    Args:
        t (Tensor): left hand side operand
        x (Tensor or float): right hand side operand

    Returns:
        a Tensor with each element being t[i] <= x ? 1.0f:0.0f,
        or t[i] <= x[i] ? 1.0f:0.0f
    '''
    return t <= x


def gt(t, x):
    '''Elementi-wise comparison for t > x.

    Args:
        t (Tensor): left hand side operand
        x (Tensor or float): right hand side operand

    Returns:
        a Tensor with each element being t[i] > x ? 1.0f:0.0f,
        or t[i] > x[i] ? 1.0f:0.0f
    '''
    return t > x


def ge(t, x):
    '''Elementi-wise comparison for t >= x.

    Args:
        t (Tensor): left hand side operand
        x (Tensor or float): right hand side operand

    Returns:
        a Tensor with each element being t[i] >= x ? 1.0f:0.0f,
        or t[i] >= x[i] ? 1.0f:0.0f
    '''
    return t >= x


def add(lhs, rhs, ret=None):
    '''Elementi-wise addition.

    Args:
        lhs (Tensor)
        rhs (Tensor)
        ret (Tensor, optional): if not None, the result is stored in it;
            otherwise, a new Tensor would be created for the result.

    Returns:
        the result Tensor
    '''
    if ret is None:
        # call Tensor.__add__()
        return lhs + rhs
    else:
        if isinstance(rhs, Tensor):
            singa.Add(lhs.singa_tensor, rhs.singa_tensor, ret.singa_tensor)
        else:
            singa.AddFloatWithRet(lhs.singa_tensor, rhs, ret.singa_tensor)
        return ret


def sub(lhs, rhs, ret=None):
    '''Elementi-wise subtraction.

    Args:
        lhs (Tensor)
        rhs (Tensor)
        ret (Tensor, optional): if not None, the result is stored in it;
            otherwise, a new Tensor would be created for the result.

    Returns:
        the result Tensor
    '''
    if ret is None:
        # call Tensor.__sub__()
        return lhs - rhs
    else:
        if isinstance(rhs, Tensor):
            singa.Sub(lhs.singa_tensor, rhs.singa_tensor, ret.singa_tensor)
        else:
            singa.SubFloatWithRet(lhs.singa_tensor, rhs, ret.singa_tensor)
        return ret


def eltwise_mult(lhs, rhs, ret=None):
    '''Elementi-wise multiplication.

    Args:
        lhs (Tensor)
        rhs (Tensor)
        ret (Tensor, optional): if not None, the result is stored in it;
            otherwise, a new Tensor would be created for the result.

    Returns:
        the result Tensor
    '''

    if ret is None:
        # call Tensor.__mul__()
        return lhs * rhs
    else:
        if isinstance(rhs, Tensor):
            singa.EltwiseMult(lhs.singa_tensor, rhs.singa_tensor,
                              ret.singa_tensor)
        else:
            singa.EltwiseMultFloatWithRet(lhs.singa_tensor, rhs,
                                          ret.singa_tensor)
        return ret


def mult(A, B, C=None, alpha=1.0, beta=0.0):
    '''Do matrix-matrix or matrix-vector multiplication.

    This function returns C = alpha * A * B + beta * C

    Args:
        A (Tensor): 2d Tensor
        B (Tensor): If B is a 1d Tensor, GEMV would be invoked for matrix-vector
            multiplication; otherwise GEMM would be invoked.
        C (Tensor, optional): for storing the result; If None, a new Tensor
            would be created.
        alpha (float)
        beta (float)

    Returns:
        the result Tensor
    '''
    if C is None:
        return _call_singa_func(singa.Mult, A.singa_tensor, B.singa_tensor)
    else:
        singa.MultWithScale(alpha, A.singa_tensor, B.singa_tensor,
                            beta, C.singa_tensor)
        return C


def tensordot(A, B, axes=2):
    """Returns the tensor multiplication of two tensors along specified axes.

    This is equivalent to compute dot product along the specified axes which
    are treated as one axis by reshaping.

    Args:
        A (Singa.Tensor): The first argument.
        B (Singa.Tensor): The second argument.
        axes:
            - If it is an integer, then ''axes'' represent axes at the last of ''a`'' and
              the first of ''b'' are used.
            - If it is a pair of sequences of integers, then these two
              sequences specify the list of axes for ''a'' and ''b''. The
              corresponding axes are paired for sum-product.

    Return:
        singa.tensor: The tensor  product of ''a'' and ''b'' along the
        axes specified by ''axes''.

    Thanks to numpy.tensordot.
    the link is https://github.com/numpy/numpy/blob/v1.14.0/numpy/core/numeric.py#L1123-L1306
    """
    # when axes is an integer, axes_A and axes_B represent axes at the last of ''a`'' and
    # the first of ''b''. For example, when axes is 1, we do the normal multiplication :
    # if A is in shape(3,2,4), B is in shape(4,2,5), it will return a matrix in shape(3,2,2,5)
    # when axes is 2 and A,B are in the shape (3,2,4) and (2,4,5), it will return a matrix in shape(3,5)
    if type(axes) == int:
        axes_A = list(range(-axes, 0))
        axes_B = list(range(0, axes))
        axes_B = axes_B
    else:
        axes_A,axes_B =axes
    # when axes is a pair of sequences of integers.For example, A is in shape(3,2,4),
    #B is in shape(4,2,5), we set axes as ([1,2],[1,0]), it will return a matrix in shape(3,5)
    if isinstance(axes_A,list):
        na = len(axes_A)
        axes_A = list(axes_A)
    else:
        axes_A = [axes_A]
        na = 1
    if isinstance(axes_B,list):
        nb = len(axes_B)
        axes_B = list(axes_B)
    else:
        axes_B = [axes_B]
        nb = 1

    # a_shape and b_shape are the shape of tensor A and B, while nda and ndb are the dim of A and B
    a_shape = A.shape
    nda = A.ndim
    b_shape = B.shape
    ndb = B.ndim
    equal = True
    # to check if the length of axe_A is equal to axes_B
    if na != nb:
        equal = False
    else:
    # to make the shape match
        for k in range(na):
            if a_shape[axes_A[k]] != b_shape[axes_B[k]]:
                equal = False
                break
            if axes_A[k] < 0:
                axes_A[k] += nda
            if axes_B[k] < 0:
                axes_B[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")
    '''start to do the calculation according to the axes'''

    notin = [k for k in range(nda) if k not in axes_A]
    # nda is the dim of A, and axes_a is the axis for A, notin is the axis which is not in axes_A
    newaxes_a = notin + axes_A
    N2 = 1
    for axis in axes_A:
        N2 *= a_shape[axis]
    N1 = 1
    for ax in notin:
        N1 *=a_shape[ax]
    #newshape_a is the shape to do multiplication.For example, A is in shape(3,2,4),
    #B is in shape(4,2,5), we set axes as ([1,2],[1,0]), then newshape_a should be (3,5)
    #olda is the shape that will be shown in the result.
    newshape_a = (N1,N2)
    olda = [a_shape[axis] for axis in notin]
    notin = [k for k in range(ndb) if k not in axes_B]
    newaxes_b = axes_B + notin
    N2 = 1
    for axis in axes_B:
        N2 *= b_shape[axis]
    N1 = 1
    for bx in notin:
        N1 *= b_shape[bx]
    newshape_b = (N2, N1)
    oldb = [b_shape[axis] for axis in notin]
    # do transpose and reshape to get the 2D matrix to do multiplication
    at = A.transpose(newaxes_a).reshape(newshape_a)
    bt = B.transpose(newaxes_b).reshape(newshape_b)
    res = mult(at, bt)
    #reshape the result
    return res.reshape(olda + oldb)


def einsum_(A,B,ops):
    '''Do the matrix to matrix einsum calculation according to the operands

    Args:
        A (Singa.Tensor): The first argument.
        B (Singa.Tensor): The second argument.
        ops(string):
            the string specifies the subscripts for summation such as 'ki,kj->kij'

    Returns: Singa.Tensor
        the output matirx of the einsum calculation
    Thanks to nils-werner/sparse.einsum(),
    the link is https://github.com/nils-werner/sparse/commit/449c75d21d3158630bc5be79c690a60cfc002578
    '''

    if len(ops) == 0:
        raise ValueError("No input operands")
    # to get the input and output ops
    inputops, outputops = ops.split('->')
    inputops = inputops.split(',')

    if A.ndim != len(inputops[0]) or B.ndim != len(inputops[1]):
        raise ValueError("input dim doesn't match operands")

    # All indices that are in input AND in output are multiplies
    # All indices that are in input BUT NOT in output are sum contractions
    multiplies = sorted(list(set(inputops[0]) & set(inputops[1]) & set(outputops)))
    sums = sorted(list((set(inputops[0]) & set(inputops[1])) - set(outputops)))

    # Map sums and indices to axis integers
    multiplies = [[inops.find(x) for x in multiplies] for inops in inputops]
    sums = [[inops.find(x) for x in sums] for inops in inputops]


    # Find output axes in input axes for final transpose
    transpose = [''.join(inputops).find(x) for x in outputops]
    #to make the transpose match to its rank
    transpose = sorted(range(len(transpose)), key = transpose.__getitem__)

    return tensormult(A,B,sums,multiplies).transpose(transpose)


def tensormult(A, B, sum=None, multiply=None):
    '''
    Args:
        A (Singa.Tensor): The first input tensor from einsum.
        B (Singa.Tensor): The second input tensor from einsum.
        sum (list[list[int]]):
            The axis to do the normal tensordot calculation

        multiply (list[list[int]]):
            The axis to multiply

    Return: (Singa.Tensor)
        The output of the tensormult calculation

    fix some error of source code in the nils-werner/sparse
    '''
    if sum is None:
        sum = [[], []]
    else:
        sum = list(sum)

    if multiply is None:
        multiply = [[], []]
    else:
        multiply = list(multiply)
    # For each multiply[0] we are adding one axis, thus we need to increment
    # all following items by one: (0, 1, 2) -> (0, 2, 4)
    idx = multipliessort(multiply[0])

    post_multiply = multiply[0]
    for i, v in enumerate(idx):
        post_multiply[v] += i


#    for i in post_multiply:
#        A = diag(A,i)
    for i in sorted(post_multiply):

        A = diag(A,i)
        sum[0] = [x+1 if x >= i else x for x in sum[0]]

    sum[0] += post_multiply
    sum[1] += multiply[1]

    return tensordot(A, B, axes=sum)

def multipliessort(multiplies):
    '''
    Returns the indices that would sort an array
    Args:
        multiplies(list[int]):
            the list of the input multiplies

    Return(list[int]):
        the indices list
    '''

    if multiplies is None:
        multiplies = []
    else:
        multiplies = list(multiplies)

    sort_multiplies = sorted(enumerate(multiplies), key=lambda x: x[1])
    idx = [x[0] for x in sort_multiplies]
    return idx

def diag(A,axis=-1):
    '''
    Return the matrix that is diagonalize along the axis
    Args:
        A (Singa.Tensor):the input tensor
        axis (int):
            the axis which the matrix use to diagonalization
    Return(Singa.Tensor):
        the tensor which has been diagonalize along the given axis


    It is like using numpy.diag() in the 1D condition, but this function can be used to do high dimension
    matrix diagonalization

    '''
    # to get the shape of the diagonalize matirx
    axis_diag = list(range(out.ndim))[axis]
    shape = list(A.shape)
    shape.insert(axis_diag,shape[axis_diag])

    npA = to_numpy(A)
    npA_diag = np.zeros(product(shape))
    index = np.argwhere(npA != 0)
    for i in index:
        i_diag = list(i)
        i_diag.insert(axis, i[axis])
        index_npA = [[i[x]] for x in range(len(i))]
        index_npA_diag = [[i_diag[x]] for x in range(len(i_diag))]
        npA_diag[index_npA_diag] = npA[index_npA]
    A_diag = from_numpy(npA_diag)
    return A_diag



def div(lhs, rhs, ret=None):
    '''Elementi-wise division.

    Args:
        lhs (Tensor)
        rhs (Tensor)
        ret (Tensor, optional): if not None, the result is stored in it;
            otherwise, a new Tensor would be created for the result.

    Returns:
        the result Tensor
    '''
    if ret is None:
        # call Tensor.__div__()
        return lhs / rhs
    else:
        if isinstance(rhs, Tensor):
            singa.Div(lhs.singa_tensor, rhs.singa_tensor, ret.singa_tensor)
        else:
            singa.DivFloatWithRet(lhs.singa_tensor, rhs, ret.singa_tensor)
        return ret


def axpy(alpha, x, y):
    '''Element-wise operation for y += alpha * x.

    Args:
        alpha (float)
        x (Tensor)
        y (Tensor)

    Returns:
        y
    '''
    singa.Axpy(float(alpha), x.singa_tensor, y.singa_tensor)
    return y


def bernoulli(p, t):
    '''Generate a binary value for each element of t.

    Args:
        p (float): each element is 1 with probability p; and 0 with 1 - p
        t (Tensor): the results are put into t

    Returns:
        t
    '''
    singa.Bernoulli(float(p), t.singa_tensor)
    return t


def gaussian(mean, std, t):
    '''Generate values following a Gaussian distribution.

    Args:
        mean (float): the mean of the Gaussian distribution.
        std (float): the standard variance of the Gaussian distribution.
        t (Tensor): the results are put into t

    Returns:
        t
    '''
    singa.Gaussian(float(mean), float(std), t.singa_tensor)
    return t


def uniform(low, high, t):
    '''Generate values following a Uniform distribution.

    Args:
        low (float): the lower bound
        hight (float): the higher bound
        t (Tensor): the results are put into t

    Returns:
        t
    '''
    singa.Uniform(float(low), float(high), t.singa_tensor)
    return t


def add_column(alpha, v, beta, M):
    '''Add v to each column of M.

    Denote each column of M as m, m = alpha * v + beta * m

    Args:
        alpha (float)
        v (Tensor)
        beta (float)
        M (Tensor): 2d tensor
    Returns:
        M
    '''
    singa.AddColumnWithScale(float(alpha), float(beta), v.singa_tensor,
                             M.singa_tensor)
    return M


def add_row(alpha, v, beta, M):
    '''Add v to each row of M.

    Denote each row of M as m, m = alpha * v + beta * m

    Args:
        alpha (float)
        v (Tensor)
        beta (float)
        M (Tensor): 2d tensor
    Returns:
        M
    '''
    singa.AddRowWithScale(alpha, beta, v.singa_tensor, M.singa_tensor)
    return M


def sum_columns(M):
    '''Sum all columns into a single column.

    Args:
        M (Tensor): the input 2d tensor.

    Returns:
        a new Tensor as the resulted column.
    '''
    assert M.ndim() == 2, 'M.nDim() is supposed to be 2'
    ret = Tensor((M.shape[0], 1), M.singa_tensor.device())
    singa.SumColumns(M.singa_tensor, ret.singa_tensor)
    return ret


def sum_rows(M):
    '''Sum all rows into a single row.

    Args:
        M (Tensor): the input 2d tensor.

    Returns:
        a new Tensor as the resulted row.
    '''
    assert M.ndim() == 2, 'M.nDim() is supposed to be 2'
    ret = Tensor((1, M.shape[1]), M.singa_tensor.device())
    singa.SumRows(M.singa_tensor, ret.singa_tensor)
    return ret


''' private functions, internally used
'''


def _call_singa_func(_singa_func, *args):
    ''' this function calls singa global functions that returns Tensor
        and create new python Tensor instance
        e.g., Tensor [singa_func](args...)
    '''
    new_t = Tensor()
    new_t.singa_tensor = _singa_func(*args)
    new_t.shape = tuple(new_t.singa_tensor.shape())
    new_t.device = new_t.singa_tensor.device()
    new_t.dtype = new_t.singa_tensor.data_type()
    return new_t
