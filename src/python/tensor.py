#!/usr/bin/env python

# /************************************************************
# *
# * Licensed to the Apache Software Foundation (ASF) under one
# * or more contributor license agreements.  See the NOTICE file
# * distributed with this work for additional information
# * regarding copyright ownership.  The ASF licenses this file
# * to you under the Apache License, Version 2.0 (the
# * "License"); you may not use this file except in compliance
# * with the License.  You may obtain a copy of the License at
# *
# *   http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing,
# * software distributed under the License is distributed on an
# * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# * KIND, either express or implied.  See the License for the
# * specific language governing permissions and limitations
# * under the License.
# *
# *************************************************************/

'''
This script includes Tensor class and its methods for python users
to call singa::Tensor and its methods
'''
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),
                             '../../build/lib'))
sys.path.append(os.path.join(os.path.dirname(__file__),
                             '../../build/python'))
import singa

sys.path.append(os.path.join(os.path.dirname(__file__),
                             '../../build/src'))
from core_pb2 import *


class Tensor(object):
    ''' Class and member functions for singa::Tensor
    '''

    def __init__(self, shape=None, device=None, dtype=kFloat32):
        ''' shape = (tuple)
        '''
        if shape is None:
            # call constructor of singa::Tensor
            self.singa_tensor = singa.Tensor()
            return
        else:
            assert type(shape) == tuple, 'shape should be tuple'
            vs = _tuple_to_vector(shape)
            if device is None:
                self.singa_tensor = singa.Tensor(vs, dtype)
            else:
                self.singa_tensor = singa.Tensor(vs, device, dtype)
            self.tuple_shape = shape
            self.device = device
            self.dtype = dtype

    def to_array(self):
        # TODO(chonho): depreciated (will be deleted later)
        idx = self.singa_tensor.data_type()
        if idx == kFloat32:
            data_array = singa.floatArray_frompointer(
                             self.singa_tensor.floatData())
            dt = np.float32
        elif idx == kFloat16:
            print 'not implemented yet'
            return
        elif idx == kInt:
            data_array = singa.intArray_frompointer(
                             self.singa_tensor.intData())
            dt = np.int32
        elif idx == kChar:
            data_array = singa.charArray_frompointer(
                             self.singa_tensor.charData())
            dt = np.int8
        elif idx == kDouble:
            data_array = singa.doubleArray_frompointer(
                             self.singa_tensor.doubleData())
            dt = np.float64

        data = [data_array[i] for i in range(self.singa_tensor.Size())]
        data = np.array(data, dtype=dt).reshape(self.tuple_shape)
        return data

    def to_numpy(self):
        ''' this method gets the values of tensor data and
            returns it as numpy array
        '''
        if self.dtype == kFloat32:
            np_array = self.singa_tensor.floatGetValue(int(self.size()))
        else:
            print 'Not implemented yet for ', self.dtype
        return np_array.reshape(self.tuple_shape)

    def copy_from_numpy(self, np_array, offset=0):
        ''' this method stores the values of numpy array into tensor data
            from the position of offset
        '''
        assert np_array.size == self.size(), 'tensor shape should be the same'
        if not np_array.ndim == 1:
            np_array = np_array.flatten()
        dt = np_array.dtype
        if dt == np.float32:
            self.singa_tensor.floatCopyDataFromHostPtr(np_array, offset)
        else:
            print 'Not implemented yet for ', dt

    def data_type(self):
        return self.singa_tensor.data_type()

    def shape(self, axis=None):
        if axis is None:
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

    def nrm2(self):
        self.singa_tensor.L2()

    def set_value(self, x):
        if type(x) == float:
            self.singa_tensor.floatSetValue(x)

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

    def bernoulli(self, p):
        if type(p) == float:
            singa.floatBernoulli(p, self.singa_tensor)

    def gaussian(self, mean, std):
        if type(mean) == float:
            singa.floatGaussian(mean, std, self.singa_tensor)

    def uniform(self, low, high):
        if type(low) == float:
            singa.floatUniform(low, high, self.singa_tensor)

    def add_column(self, v):
        singa.AddColumn(v.singa_tensor, self.singa_tensor)

    def add_row(self, v):
        singa.AddRow(v.singa_tensor, self.singa_tensor)

    def div_column(self, v):
        singa.DivColumn(v.singa_tensor, self.singa_tensor)

    def div_row(self, v):
        singa.DivRow(v.singa_tensor, self.singa_tensor)

    def mult_column(self, v):
        singa.MultColumn(v.singa_tensor, self.singa_tensor)

    def mult_row(self, v):
        singa.MultRow(v.singa_tensor, self.singa_tensor)

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
            return _call_singa_func(singa.Add_TT,
                                    self.singa_tensor, rhs.singa_tensor)
        else:
            return _call_singa_func(singa.Add_Tf,
                                    self.singa_tensor, rhs)

    def __sub__(self, rhs):
        if isinstance(rhs, Tensor):
            return _call_singa_func(singa.Sub_TT,
                                    self.singa_tensor, rhs.singa_tensor)
        else:
            return _call_singa_func(singa.Sub_Tf,
                                    self.singa_tensor, rhs)

    def __mul__(self, rhs):
        if isinstance(rhs, Tensor):
            return _call_singa_func(singa.EltwiseMul_TT,
                                    self.singa_tensor, rhs.singa_tensor)
        else:
            return _call_singa_func(singa.EltwiseMul_Tf,
                                    self.singa_tensor, rhs)

    def __div__(self, rhs):
        if isinstance(rhs, Tensor):
            return _call_singa_func(singa.Div_TT,
                                    self.singa_tensor, rhs.singa_tensor)
        else:
            return _call_singa_func(singa.Div_Tf,
                                    self.singa_tensor, rhs)

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


def copy_data_to_from(dst, src, size, dst_offset=0, src_offset=0):
    singa.CopyDataToFrom(dst.singa_tensor, src.singa_tensor, size,
                         dst_offset, src_offset)


def from_numpy(np_array):
    ret = Tensor(np_array.shape)
    ret.copy_from_numpy(np_array)
    return ret


def to_numpy(t):
    return t.to_numpy()


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
    if axis is None:
        return singa.floatSum(t.singa_tensor)
    else:
        return _call_singa_func(singa.Sum, t.singa_tensor, axis)


def pow(t, x, out=None):
    if out is None:
        if isinstance(x, Tensor):
            return _call_singa_func(singa.Pow, t.singa_tensor, x.singa_tensor)
        else:
            return _call_singa_func(singa.Pow_f, t.singa_tensor, x)
    else:
        if isinstance(x, Tensor):
            singa.Pow(t.singa_tensor, x.singa_tensor, out.singa_tensor)
        else:
            singa.Pow_f_out(t.singa_tensor, x, out.singa_tensor)
        return out


def average(t, axis=0):
    return _call_singa_func(singa.Average, t.singa_tensor, axis)


def softmax(t, out=None):
    if out is None:
        return _call_singa_func(singa.SoftMax, t.singa_tensor)
    else:
        singa.SoftMax(t.singa_tensor, out.singa_tensor)
        return out


def lt(t, x):
    return t < x


def le(t, x):
    return t <= x


def gt(t, x):
    return t > x


def ge(t, x):
    return t >= x


def add(lhs, rhs, ret=None):
    if ret is None:
        # call Tensor.__add__()
        return lhs + rhs
    else:
        if isinstance(rhs, Tensor):
            singa.Add(lhs.singa_tensor, rhs.singa_tensor, ret.singa_tensor)
        else:
            singa.Add_Tf_out(lhs.singa_tensor, rhs, ret.singa_tensor)
        return ret

def sub(lhs, rhs, ret=None):
    if ret is None:
        # call Tensor.__sub__()
        return lhs - rhs
    else:
        if isinstance(rhs, Tensor):
            singa.Sub(lhs.singa_tensor, rhs.singa_tensor, ret.singa_tensor)
        else:
            singa.Sub_Tf_out(lhs.singa_tensor, rhs, ret.singa_tensor)
        return ret


def eltwise_mult(lhs, rhs, ret=None):
    if ret is None:
        # call Tensor.__mul__()
        return lhs * rhs
    else:
        if isinstance(rhs, Tensor):
            singa.EltwiseMult(lhs.singa_tensor, rhs.singa_tensor,
                              ret.singa_tensor)
        else:
            singa.EltwiseMult_Tf_out(lhs.singa_tensor, rhs,
                                     ret.singa_tensor)
        return ret


def mult(A, B, C=None, alpha=1.0, beta=0.0):
    '''
    This function returns C = alpha * A * B + beta * C
    '''
    if C is None:
        return _call_singa_func(singa.Mult, A.singa_tensor, B.singa_tensor)
    else:
        singa.floatMult(alpha, A.singa_tensor, B.singa_tensor,
                        beta, C.singa_tensor)
        return C


def div(lhs, rhs, ret=None):
    if ret is None:
        # call Tensor.__div__()
        return lhs / rhs
    else:
        if isinstance(rhs, Tensor):
            singa.Div(lhs.singa_tensor, rhs.singa_tensor, ret.singa_tensor)
        else:
            singa.Div_Tf_out(lhs.singa_tensor, rhs, ret.singa_tensor)
        return ret


def axypbz(alpha, A, B, b, C):
    # TODO(chonho): depreciated (will be deleted later)
    singa.floatMult(alpha, A.singa_tensor, B.singa_tensor, b, C.singa_tensor)
    return C


def axpy(alpha, x, y):
    if type(alpha) == float:
        singa.floatAxpy(alpha, x.singa_tensor, y.singa_tensor)
    return y


def bernoulli(p, t):
    if type(p) == float:
        singa.floatBernoulli(p, t.singa_tensor)
    return t


def gaussian(mean, std, t):
    if type(mean) == float:
        singa.floatGaussian(mean, std, t.singa_tensor)
    return t


def uniform(low, high, t):
    if type(low) == float:
        singa.floatUniform(low, high, t.singa_tensor)
    return t


def add_column(alpha, v, beta, M):
    singa.floatAddColumn(alpha, beta, v.singa_tensor, M.singa_tensor)
    return M


def add_row(alpha, v, beta, M):
    singa.floatAddRow(alpha, beta, v.singa_tensor, M.singa_tensor)
    return M


def sum_columns(M):
    assert M.ndim() == 2, 'M.nDim() is supposed to be 2'
    nb_col = M.shape(0)
    ret = Tensor((nb_col, 1))
    singa.SumColumns(M.singa_tensor, ret.singa_tensor)
    return ret


def sum_rows(M):
    assert M.ndim() == 2, 'M.nDim() is supposed to be 2'
    nb_row = M.shape(1)
    ret = Tensor((1, nb_row))
    singa.SumRows(M.singa_tensor, ret.singa_tensor)
    return ret


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
