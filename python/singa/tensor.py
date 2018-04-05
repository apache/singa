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
        data: a singa_tensor recording input data.
        creator: a Operation object which generate this tensor.
        requires_grad: a bool recording if the creator of tensor require gradient.
        grad_outlet: a bool recording if the tensor is a outlet for gradient.

    '''
    def __init__(self, shape=None, device=None, dtype=core_pb2.kFloat32, data=None, creator=None, requires_grad=True,
                 grad_outlet=False):
        if shape is None:
            # call constructor of singa::Tensor
            self.singa_tensor = singa.Tensor()
        else:
            assert isinstance(shape, tuple), 'shape should be tuple'
            if device is None:
                device = pydevice.get_default_device()
                self.singa_tensor = singa.Tensor(list(shape), device, dtype)
            else:
                self.singa_tensor = singa.Tensor(list(shape), device, dtype)
        if data is not None:
            self.singa_tensor = data
            if creator is None:
                creator = Initializer(self, requires_grad)

        self.shape = tuple(self.singa_tensor.shape())
        self.device = self.singa_tensor.device()
        self.dtype = self.singa_tensor.data_type()

        self.creator = creator
        self.grad_outlet = grad_outlet

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
            self.singa_tensor /= float(x)
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


def copy_from_numpy(singa_tensor, np_array):
    '''
    Copy the data from the numpy array.
    '''
    assert np_array.size == singa_tensor.Size(), 'tensor shape should be the same'
    if not np_array.ndim == 1:
        np_array = np_array.flatten()
    dt = np_array.dtype
    if dt == np.float32:
        singa_tensor.CopyFloatDataFromHostPtr(np_array)
    elif dt == np.int or dt == np.int32:
        singa_tensor.CopyIntDataFromHostPtr(np_array)
    else:
        print('Not implemented yet for ', dt)


class Operation(object):
    '''
    Wrap normal functions such as dot to realize autograd.

    '''
    def __init__(self, **operation_params):
        pass

    def __call__(self, *input):
        return self._do_forward(*input)

    def _do_forward(self, *input):
        unpacked_input = tuple(arg.singa_tensor for arg in input)
        raw_output = self.forward(*unpacked_input)
        if not isinstance(raw_output, tuple):
            raw_output = (raw_output,)
        self.needs_input_grad = tuple(arg.creator.requires_grad for arg in input)
        self.requires_grad = any(self.needs_input_grad)
        output = tuple(Tensor(data=data, creator=self) for data in raw_output)
        self.previous_functions = [(arg.creator, id(arg)) for arg in input]
        self.output_ids = {id(var): i for i, var in enumerate(output)}
        return output

    def _do_backward(self, grad_output):
        grad_input = self.backward(grad_output)
        if not isinstance(grad_input, tuple):
            grad_input = (grad_input,)
        return grad_input

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *grad_output):
        raise NotImplementedError


class Initializer(Operation):
    '''
    For Tensor without creator, Initializer can act as its creator.
    It is commonly used in feeding training data or initialize parameters like weights and bias.

    '''
    def __init__(self, Tensor, requires_grad):
        self.Tensor = Tensor
        self.output_ids = {id(Tensor): 0}
        self.previous_functions = []
        self.requires_grad = requires_grad
        shape = self.Tensor.singa_tensor.shape()
        self.init = singa.Tensor(list(shape))
        copy_from_numpy(self.init, np.zeros(shape=shape, dtype=np.float32))
        self.grads = self.init.Clone()

    def _do_forward(self):
        raise NotImplementedError

    def _do_backward(self, *dy):
        assert len(dy) == 1
        self.grads = singa.__add__(self.grads, dy[0])
        return tuple()


class ReLU(Operation):
    def forward(self, x):
        '''
        forward function for ReLU Operation.

        '''
        self.input = (x,)
        return singa.ReLU(x)

    def backward(self, dy):
        '''
        backward function for ReLU Operation.
        '''
        dx = singa.GTFloat(self.input[0], 0.0)
        return singa.__mul__(dy, dx)
def relu(x):
    return ReLU()(x)[0]


class Dot(Operation):
    def forward(self, x, w):
        '''
        forward function for Dot Operation.

        '''
        self.input = (x, w)
        return singa.Mult(x, w)

    def backward(self, dy):
        '''
        backward function for Dot Operation.

        '''
        return singa.Mult(dy, self.input[1].T()), singa.Mult(self.input[0].T(), dy)
def dot(x, w):
    return Dot()(x, w)[0]


class Add_Bias(Operation):
    def forward(self, b, x):
        '''
        forward function for Add_Bias Operation.

        '''
        singa.AddRow(b, x)
        return x

    def backward(self, dy):
        '''
        backward function for Add_Bias Operation.

        '''
        return singa.Sum(dy, 0), dy
def add_bias(b, x):
    return Add_Bias()(b, x)[0]


class SoftMax(Operation):
    def forward(self, x):
        '''
        forward function for SoftMax Operation.

        '''
        self.output = (singa.SoftMax(x),)
        return self.output[0]

    def backward(self, dy):
        '''
        backward function for SoftMax Operation.

        '''
        # calculations are made on numpy
        grad = To_Numpy(dy)
        output = To_Numpy(self.output[0])
        out_1 = np.einsum('ki,ki->ki', grad, output)
        medium_out = np.einsum('ki,kj->kij', output, output)
        out_2 = np.einsum('kij,kj->ki', medium_out, grad)
        out = out_1 - out_2
        out_singa = singa.Tensor(out_1.shape)
        out_singa.CopyFloatDataFromHostPtr(out.flatten())
        return out_singa
def softmax(x):
    return SoftMax()(x)[0]


class Cross_Entropy(Operation):
    def forward(self, pred, target):
        '''
        forward function for Cross_Entropy Operation.

        '''
        loss = singa.Tensor((1,))
        loss.SetFloatValue(-singa.SumAsFloat(singa.__mul__(target, singa.Log(pred)))/pred.shape()[0])
        self.input = (pred, target)
        return loss

    def backward(self, dy):
        '''
        backward function for Cross_Entropy Operation.

        '''
        dx = singa.__div__(self.input[1], self.input[0])
        dx *= float(-1/self.input[0].shape()[0])
        if not isinstance(dy, singa.Tensor):
            # dtype of dy: float
            dx *= dy
            return dx
        else:
            pass  # TODO
def cross_entropy(y, t):
    return Cross_Entropy()(y, t)[0]


def To_Numpy(x):
    '''
    To be used in SoftMax Operation.
    Convert a singa_tensor to numpy_tensor.
    '''
    np_array = x.GetFloatValue(int(x.Size()))
    return np_array.reshape(x.shape())