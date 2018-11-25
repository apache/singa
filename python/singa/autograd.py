#
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
#

from __future__ import division

from collections import Counter, deque
import numpy as np
import math

from .tensor import Tensor
from . import layer
from singa.proto import model_pb2
from . import singa_wrap as singa
#from .tensor import einsum

CTensor = singa.Tensor
training = False






def infer_dependency(op):
    '''
    Infer the dependency of all operations with the
    given op as the last operation.
    Operation A is depending on B is A uses the output(s) of B.
    Args:
        op: an Operation instance, e.g. the loss operation.
    Return:
        a Counter instance with the operation as the key,
        and the number of operations that are depending on it as the value
    '''
    # not count the dependency of current op.
    # if the current op is not a terminal op, then this function may just
    # count dependency of a branch.
    dependency_count = Counter()
    queue = deque([op])
    while len(queue) > 0:
	cur_op = queue.pop()
	#print(cur_op)
        for src_op, _, _, _ in cur_op.src:
            if src_op not in dependency_count:
                # dependency[src_op] = [Counter() for _ in src_op.y_id2idx]
                if isinstance(src_op, Dummy):
                    # only when a Dummy operator needs store grads, its
                    # dependency needs to be counted.
                    if src_op.stores_grad:
                        dependency_count[src_op] = 0
                        queue.append(src_op)
                else:
                    dependency_count[src_op] = 0
                    queue.append(src_op)
            # y_idx = src_op.y_id2idx[x_id]
            # dependency[src_op][y_idx][cur_op] += 1
            if dependency_count.has_key(src_op):
                dependency_count[src_op] += 1
    return dependency_count


def gradients(y, dy=None):
    grads = {}  # mapping: x->dx if x.stores_grad
    for p, dp in backward(y, dy):
        grads[p] = dp
    return grads





def backward(y, dy=None):
    '''
    Run the backward propagation starting at y.
    Args:
        y: a Tensor instance, usually the loss
        dy: a number or a Tensor instance, for the gradient of the
            objective/loss w.r.t y, usually 1.0
    Return:
        a dictionary storing the gradient tensors of all tensors
        whose stores_grad is true (e.g. parameter tensors)
    '''
    assert isinstance(y, Tensor), 'wrong input type.'
    dependency = infer_dependency(y.creator)
    assert y.size() == 1, 'y must be a Tensor with a single value;'\
        'size of y is % d' % y.size()

    # by default the dy is a tensor with 1.0 for each sample;
    if dy is None:
        dy = float(1.0)
    elif isinstance(dy, Tensor):
        dy = dy.data
    else:
        dy = float(dy)

    # ready is a queue of (operation, dy list)
    ready = deque([(y.creator, (dy,))])
    not_ready = {}  # mapping: op->[dy]

    if y.stores_grad:
        #gradients[y] = dy
        if isinstance(dy, float):
            g = np.array(dy)
        else:
            g = dy
        tg = Tensor(device=g.device(), data=g)
        yield (y, tg)

    while len(ready) > 0:
        op, dys = ready.pop()
        if not op.requires_grad or isinstance(op, Dummy):
            continue
        # if not isinstance(op, tensor.Dummy):
        dxs = op._do_backward(*dys)
	#print('backward',op)
        # TODO src and dx must match
        assert len(op.src) == len(dxs), \
            'the number of src ops (=%d) and dx (=%d) not match' \
            % (len(op.src), len(dxs))
        for (src_op, x_id, y, y_stores_grad), dx in zip(op.src, dxs):
            # prefix x is w.r.t op; prefix y is w.r.t src_op.
            # x_id is the python id of one input arg of src_op, denoted as x.
            # y_idx (below) is the index of x among the outputs of src_op.
            # not_ready[src_op][y_idx] records the intermediate gradient
            # of the y_idx'th output of src_op. 'intermediate gradient'
            # indicates that if this output is used in multiple children
            # operations, then we have to add the graident (dx) from all these
            # children operations. When src_op is ready, it means that
            # the gradient of all its outputs are available, i.e. all children
            # operations have been backwarded.
            # y is None if y.stores_grad is false; otherwise it is a Tensor

            if isinstance(src_op, Dummy):
                if not src_op.stores_grad:
                    continue

            y_idx = src_op.y_id2idx[x_id]
            if src_op not in not_ready:
                # src_op may have mulitple outputs
                not_ready[src_op] = [None for _ in src_op.y_id2idx]
                not_ready[src_op][y_idx] = dx
            else:
                dxs = not_ready[src_op]
                if dxs[y_idx] is None:
                    dxs[y_idx] = dx
                else:
                    # add the gradient from another children operation that
                    # uses y_idx'th output of src_op as input arg
                    dxs[y_idx] += dx

            dependency[src_op] -= 1

            if y_stores_grad:
                if dependency[src_op] == 0:
                    # store the gradient for final return, e.g. if x is parameter
                    # may cause a delay output, as only after src_op is ready
                    # then output, not the current outlet of src_op is ready
                    # then output.
                    g = not_ready[src_op][y_idx]
                    tg = Tensor(device=g.device(), data=g)
                    yield (y, tg)

            if src_op.requires_grad is True:
                if dependency[src_op] == 0:
                    if not isinstance(src_op, Dummy):
                        # Dummy can be in not_ready list but cannot be in ready
                        # list.
                        ready.append((src_op, not_ready[src_op]))
                    del not_ready[src_op]
        del op  # delete the operation to free all tensors from this op


class Operation(object):
    '''
    An operation includes the forward and backward function of
    tensor calculation.
    Steps to add a specific operation Xxxx:
    1. create a subclass of Operation, name it as Xxxx
    2. override the forward() and backward(); The arguments of forward()
       and backward() should only include CTensor;
    '''

    def __call__(self, *xs):
        return self._do_forward(*xs)

    def _do_forward(self, *xs):
        '''
        Do not call this function from user code. It is called by __call__().
        Args:
            xs, Tensor instance(s)
        Returns:
            Tensor instance(s)
        '''
        # TODO add the pre hook
        assert all([isinstance(x, Tensor) for x in xs]), \
            'xs should include only Tensor instances'

        # need to do backward if any of its input arg needs gradient
        self.requires_grad = any([x.requires_grad for x in xs])

        self.src = []
        for x in xs:
            if x.stores_grad:
                # store the tensor whose gradient needs be returned in
                # backward(), e.g. if x is parameter
                self.src.append((x.creator, id(x), x, x.stores_grad))
            else:
                # for intermediate tensors, they will be released soon;
                # no need to store them --> use None
                self.src.append((x.creator, id(x), None, x.stores_grad))

        # get the CTensor (data) if the input arg is Tensor
        xs = tuple(x.data for x in xs)
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        # create Tensor based on CTensor(data);
        # assume outputs are all Tensor instances
        ys = tuple(Tensor(device=y.device(),
                          data=y,
                          requires_grad=self.requires_grad,
                          creator=self) for y in ys)
        # map from python id to output index
        self.y_id2idx = {id(y): i for i, y in enumerate(ys)}
        # TODO add the post hook
        return ys

    def _do_backward(self, *dys):
        dxs = self.backward(*dys)
        if not isinstance(dxs, tuple):
            dxs = (dxs,)
        return dxs

    def forward(self, *xs):
        '''Forward propagation.
        Args:
            xs: input args consisting of only CTensors.
        Returns:
            CTensor instance(s)
        '''
        raise NotImplementedError

    def backward(self, *dys):
        ''' Backward propagation.
        Args:
            dys: input args consisting of only CTensors.
        Returns:
            CTensor instance(s)
        '''
        raise NotImplementedError

    def get_params(self):
        return []


class Dummy(Operation):
    '''Dummy operation whice serves as a placehoder for autograd
    Args:
        name(string): set it for debug
    '''

    def __init__(self, tensor, name=None):
        self.name = name
        self.src = []
        self.y_id2idx = {id(tensor): 0}
        self.stores_grad = tensor.stores_grad
        self.requires_grad = False


class ReLU(Operation):

    def forward(self, x):
        '''
        Args:
            x(CTensor): input tensor
        Returns:
            a new CTensor whose element y = x if x >= 0; otherwise 0;
        '''
        self.param={'name':'LeakyRelu','x':x,'alpha':0.0}
	if training:
            self.input = x
        return singa.ReLU(x)

    def backward(self, dy):
        '''
        Args:
            dy(CTensor): dL / dy
        Returns:
            dx(CTensor): dL / dx = dy if x >= 0; otherwise 0;
        '''
        dx = singa.GTFloat(self.input, 0.0)
        return singa.__mul__(dy, dx)


def relu(x):
    return ReLU()(x)[0]


class Matmul(Operation):
    '''For matrix multiplication'''

    def forward(self, x, w):
        '''Do forward propgation.
        Store the x(or w) if w(or x) requires gradient.
        Args:
            x (CTensor): matrix
            w (CTensor): matrix
        Returns:
            a CTensor for the result
        '''
	self.param = {'name':'MatMul','w':w,'x':x}
        if training:
            self.input = (x, w)
        return singa.Mult(x, w)

    def backward(self, dy):
        '''
        Args:
            dy (CTensor): data for the dL / dy, L is the loss
        Returns:
            a tuple for (dx, dw)
        '''
        return singa.Mult(dy, singa.DefaultTranspose(self.input[1])), \
            singa.Mult(singa.DefaultTranspose(self.input[0]), dy)


def matmul(x, w):
    return Matmul()(x, w)[0]


class AddBias(Operation):
    '''
    Add Bias to each row / column of the Tensor, depending on the axis arg.
    '''

    def __init__(self, axis=0):
        '''
        To indicate the calculation axis, 0 for row, 1 for column.
        Args:
            axis: 0 or 1, default is 0.
        '''
	self.param = {'name':'Add','axis':axis}
        self.axis = axis

    def forward(self, x, b):
        '''
        Args:
            x: matrix.
            b: bias to be added.
        Return:
            the result Tensor
        '''
	self.param['x'] = x
        self.param['b'] = b
        if self.axis == 0:
            singa.AddRow(b, x)
        elif self.axis == 1:
            singa.AddColumn(b, x)
        return x

    def backward(self, dy):
        '''
        Args:
            dy (CTensor): data for the dL / dy, L is the loss.
        Return:
            a tuple for (db, dx), db is data for dL / db, dx is data
            for dL / dx.
        '''
        if self.axis == 0:
            return dy, singa.Sum(dy, 0)
        elif self.axis == 1:
            return dy, singa.Sum(dy, 0)


def add_bias(x, b, axis=0):
    return AddBias(axis)(x, b)[0]


class Add(Operation):

    def forward(self, a, b):
	self.param = {'name': 'Add'}
        return singa.__add__(a, b)

    def backward(self, dy):
        return dy, dy


def add(a, b):
    return Add()(a, b)[0]


class SoftMax(Operation):
    '''
    Apply SoftMax for each row of the Tensor or each column of the Tensor
    according to the parameter axis.
    '''

    def __init__(self, axis=0):
        self.axis = axis
	self.param={'name':'Softmax','axis':axis}

    def forward(self, x):
        '''
        Args:
            x(data): the input 1d or 2d tensor
        Returns:
            the result Tensor
        '''
	self.param['x']=x
        if self.axis == 1:
            x = singa.DefaultTranspose(x)
        self.output = singa.SoftMax(x)
        if self.axis == 0:
            return self.output
        elif self.axis == 1:
            return singa.DefaultTranspose(self.output)

    def backward(self, dy):
        '''
        Args:
            dy (CTensor): data for the dL / dy, L is the loss
        Returns:
            dx (Ctensor): data for the dL / dx, L is the loss,
            x is the input of current Opertion
        '''
        # calculations are made on numpy array
        if self.axis == 1:
            dy = singa.DefaultTranspose(dy)
        grad = ctensor2numpy(dy)
        output = ctensor2numpy(self.output)
        out_1 = np.einsum('ki,ki->ki', grad, output)
        medium_out = np.einsum('ki,kj->kij', output, output)
        out_2 = np.einsum('kij,kj->ki', medium_out, grad)
        out = out_1 - out_2
        dx = CTensor(out_1.shape)
        dx.CopyFloatDataFromHostPtr(out.flatten())
        '''grad = Tensor(data=dy)
        output = Tensor(data=self.output)
        out_1 = einsum('ki,ki->ki', grad, output)
        medium_out = einsum('ki,kj->kij', output, output)
        out_2 = einsum('kij,kj->ki', medium_out, grad)
        out = out_1 - out_2
        dx = CTensor(out_1.data.shape)
        dx.CopyFloatDataFromHostPtr(out.data.flatten())'''
        if self.axis == 0:
            return dx
        elif self.axis == 1:
            return singa.DefaultTranspose(dx)


def softmax(x, axis=0):
    return SoftMax(axis)(x)[0]


class CrossEntropy(Operation):
    '''
    Calculte negative log likelihood loss for a batch of training data.
    '''

    def forward(self, x, t):
	self.param = {'name': 'CrossEntropy'}
        '''
        Args:
            x (CTensor): 1d or 2d tensor, the prediction data(output)
                         of current network.
            t (CTensor): 1d or 2d tensor, the target data for training.
        Returns:
            loss (CTensor): scalar.
        '''
        loss = CTensor((1,))
        loss_data = -singa.SumAsFloat(singa.__mul__(t, singa.Log(x)))
        loss.SetFloatValue(loss_data / x.shape()[0])
        self.x = x
        self.t = t
        self.input = (x, t)
        return loss

    def backward(self, dy=1.0):
        '''
        Args:
            dy (float or CTensor): scalar, accumulate gradient from outside
                                of current network, usually equal to 1.0
        Returns:
            dx (CTensor): data for the dL /dx, L is the loss, x is the output
                          of current network. note that this is true for
                          dy = 1.0
        '''
        dx = singa.__div__(self.t, self.x)
        dx *= float(-1 / self.x.shape()[0])
        if isinstance(dy, float):
            # dtype of dy: float
            dx *= dy
            return dx, None
        elif isinstance(dy, CTensor):
            pass  # TODO, broadcast elementwise multiply seems not support


def cross_entropy(y, t):
    return CrossEntropy()(y, t)[0]


class SoftMaxCrossEntropy(Operation):

    def __init__(self, t):
        self.t = t.data

    def forward(self, x):
        self.p = singa.SoftMax(x)
        loss = CTensor((1,), self.p.device())
        ret = singa.CrossEntropyFwd(self.p, self.t)
        loss.SetFloatValue(singa.SumAsFloat(ret) / x.shape()[0])
        return loss

    def backward(self, dy=1.0):
        dx = singa.SoftmaxCrossEntropyBwd(self.p, self.t)
        return singa.DivFloat(dx, float(self.p.shape()[0]))


def softmax_cross_entropy(x, t):
    # x is the logits and t is the ground truth; both are 2D.
    return SoftMaxCrossEntropy(t)(x)[0]


class MeanSquareError(Operation):

    def forward(self, x, t):
        self.err = singa.__sub__(x, t)
        sqr = singa.Square(self.err)
        loss = CTensor((1,), x.device())
        loss.SetFloatValue(singa.SumAsFloat(sqr) / x.shape()[0] / 2)
        return loss

    def backward(self, dy=1.0):
        dx = self.err
        dx *= float(1 / self.err.shape()[0])
        if isinstance(dy, float):
            # dtype of dy: float
            dx *= dy
            return dx, None
        elif isinstance(dy, CTensor):
            pass  # TODO, broadcast elementwise multiply seems not support


def mse_loss(x, t):
    return MeanSquareError()(x, t)[0]


def ctensor2numpy(x):
    '''
    To be used in SoftMax Operation.
    Convert a singa_tensor to numpy_tensor.
    '''
    np_array = x.GetFloatValue(int(x.Size()))
    return np_array.reshape(x.shape())


class Flatten(Operation):

    def __init__(self, start_axis=1):
        # flatten all axis after (inclusive) start_axis
	self.param={'name':'Flatten'}
        self.start_axis = start_axis
        assert start_axis == 1, 'must flatten into 2d array not'

    def forward(self, x):
        # TODO Do flatten start from axis != 1
        self.shape = list(x.shape())
        y = singa.Reshape(x, (x.shape()[0], x.Size() // x.shape()[0]))
        return y

    def backward(self, dy):
        dx = singa.Reshape(dy, self.shape)
        return dx


def flatten(x):
    return Flatten()(x)[0]


class Layer(object):

    def __init__(self):
        pass

    def device_check(self, *inputs):
        x_device = inputs[0].device
        for var in inputs:
            if var.device.id() != x_device:
                var.to_device(x_device)


class Linear(Layer):

    def __init__(self, in_features, out_features, bias=True):
        w_shape = (in_features, out_features)
        b_shape = (1, out_features)
        self.bias = bias

        self.W = Tensor(shape=w_shape,
                        requires_grad=True, stores_grad=True)
        std = math.sqrt(2.0 / (in_features + out_features))
        self.W.gaussian(0.0, std)

        if self.bias:
            self.b = Tensor(shape=b_shape,
                            requires_grad=True, stores_grad=True)
            self.b.set_value(0.0)

    def __call__(self, x):
        if self.bias:
            self.device_check(x, self.W, self.b)
        else:
            self.device_check(x, self.W)
        y = matmul(x, self.W)
        if self.bias:
            y = add_bias(y, self.b, axis=0)
        return y


class Concat(Operation):

    def __init__(self, axis=0):
        self.axis = axis

    def forward(self, *xs):
        if training:
            offset = 0
            self.slice_point = []
            for t in xs:
                offset += t.shape()[self.axis]
                self.slice_point.append(offset)
        x = singa.VecTensor(list(xs))
        return singa.ConcatOn(x, self.axis)

    def backward(self, dy):
        assert hasattr(
            self, 'slice_point'), 'Please set training as True before do BP. '
        assert self.slice_point[-1] == dy.shape()[self.axis], 'Shape dismatched.'
        dxs = []
        last_offset = 0
        for p in self.slice_point:
            dxs.append(singa.SliceOn(dy, last_offset, p, self.axis))
            last_offset = p
        return tuple(dxs)


def cat(xs, axis=0):
    # xs is a tuple of multiple Tensors
    return Concat(axis)(*xs)[0]


class _Conv2d(Operation):

    def __init__(self, handle):
        self.handle = handle

    def forward(self, x, W, b):
        assert x.nDim() == 4, 'The dimensions of input should be 4D.'

        if training:
            if self.handle.bias_term:
                self.inputs = (x, W, b)
            else:
                self.inputs = (x, W)

        if self.handle.device_id == -1:
            return singa.CpuConvForward(x, W, b, self.handle)

        else:
            return singa.GpuConvForward(x, W, b, self.handle)

    def backward(self, dy):
        assert training is True and hasattr(
            self, 'inputs'), 'Please set training as True before do BP. '

        if dy.device().id() != self.handle.device_id:
            dy.ToDevice(self.inputs[0].device())

        if self.handle.device_id == -1:
            dx = singa.CpuConvBackwardx(
                dy, self.inputs[1], self.inputs[0], self.handle)
            dW = singa.CpuConvBackwardW(
                dy, self.inputs[0], self.inputs[1], self.handle)
            if self.handle.bias_term:
                db = singa.CpuConvBackwardb(dy, self.inputs[2], self.handle)
                return dx, dW, db
            else:
                return dx, dW, None
        else:
            dx = singa.GpuConvBackwardx(
                dy, self.inputs[1], self.inputs[0], self.handle)
            dW = singa.GpuConvBackwardW(
                dy, self.inputs[0], self.inputs[1], self.handle)
            if self.handle.bias_term:
                db = singa.GpuConvBackwardb(dy, self.inputs[2], self.handle)
                return dx, dW, db
            else:
                return dx, dW, None


def conv2d(handle, x, W, b):
    return _Conv2d(handle)(x, W, b)[0]


class Conv2d(Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.groups = groups

        assert self.groups >= 1 and self.in_channels % self.groups == 0, 'please set reasonable groups.'

        # each group should contribute equally to the output feature maps. shown as the later part of
        # the following judgement.
        assert self.out_channels >= self.groups and self.out_channels % self.groups == 0, 'out_channels and groups dismatched.'

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise TypeError('Wrong kernel_size type.')

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
        else:
            raise TypeError('Wrong stride type.')

        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple):
            self.padding = padding
        else:
            raise TypeError('Wrong padding type.')

        if dilation != 1:
            raise ValueError('Not implemented yet')

        self.bias = bias

        self.inner_params = {'cudnn_prefer': 'fastest',
                             'workspace_MB_limit': 1024}
        # TODO valid value of inner_params check

        for kwarg in kwargs:
            if kwarg not in self.inner_params:
                raise TypeError('Keyword argument not understood:', kwarg)
            else:
                self.inner_params[kwarg] = kwargs[kwarg]

        w_shape = (self.out_channels, int(self.in_channels / self.groups),
                   self.kernel_size[0], self.kernel_size[1])

        self.W = Tensor(shape=w_shape, requires_grad=True, stores_grad=True)
        # std = math.sqrt(
        # 2.0 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1] +
        # self.out_channels))
        std = math.sqrt(
            2.0 / (w_shape[1] * self.kernel_size[0] * self.kernel_size[1] + self.out_channels))
        self.W.gaussian(0.0, std)

        if self.bias:
            b_shape = (self.out_channels,)
            self.b = Tensor(shape=b_shape, requires_grad=True,
                            stores_grad=True)
            self.b.set_value(0.0)
        else:
            # to keep consistency when to do forward.
            self.b = Tensor(data=CTensor(
                []), requires_grad=False, stores_grad=False)

    def __call__(self, x):
        assert x.shape[1] == self.in_channels, 'in_channels dismatched'

        self.device_check(x, self.W, self.b)

        if x.device.id() == -1:
            if self.groups != 1:
                raise ValueError('Not implemented yet')
            else:
                if not hasattr(self, 'handle'):
                    self.handle = singa.ConvHandle(x.data, self.kernel_size, self.stride,
                                                   self.padding, self.in_channels, self.out_channels, self.bias)
                elif x.shape[0] != self.handle.batchsize:
                    self.handle = singa.ConvHandle(x.data, self.kernel_size, self.stride,
                                                   self.padding, self.in_channels, self.out_channels, self.bias)
        else:
            if not hasattr(self, 'handle'):
                self.handle = singa.CudnnConvHandle(x.data, self.kernel_size, self.stride,
                                                    self.padding, self.in_channels, self.out_channels, self.bias, self.groups)
            elif x.shape[0] != self.handle.batchsize:
                self.handle = singa.CudnnConvHandle(x.data, self.kernel_size, self.stride,
                                                    self.padding, self.in_channels, self.out_channels, self.bias, self.groups)
        self.handle.device_id = x.device.id()

        y = conv2d(self.handle, x, self.W, self.b)
        return y


class SeparableConv2d(Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):

        self.spacial_conv = Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)

        self.depth_conv = Conv2d(in_channels, out_channels, 1, bias=bias)

    def __call__(self, x):
        y = self.spacial_conv(x)
        y = self.depth_conv(y)
        return y


class BatchNorm2d(Layer):

    def __init__(self, num_features, momentum=0.9):
        self.channels = num_features
        self.momentum = momentum

        param_shape = (self.channels,)

        self.scale = Tensor(shape=param_shape,
                            requires_grad=True, stores_grad=True)
        self.scale.set_value(1.0)

        self.bias = Tensor(shape=param_shape,
                           requires_grad=True, stores_grad=True)
        self.bias.set_value(0.0)

        self.running_mean = Tensor(
            shape=param_shape, requires_grad=False, stores_grad=False)
        self.running_var = Tensor(
            shape=param_shape, requires_grad=False, stores_grad=False)

    def __call__(self, x):
        assert x.shape[1] == self.channels, 'number of channels dismatched. %d vs %d' % (
            x.shape[1], self.channels)

        self.device_check(x, self.scale, self.bias,
                          self.running_mean, self.running_var)

        if x.device.id() == -1:
            raise NotImplementedError

        else:
            if not hasattr(self, 'handle'):
                self.handle = singa.CudnnBatchNormHandle(
                    self.momentum, x.data)
            elif x.shape[0] != self.handle.batchsize:
                self.handle = singa.CudnnBatchNormHandle(
                    self.momentum, x.data)
        self.handle.device_id = x.device.id()

        y = batchnorm_2d(self.handle, x, self.scale, self.bias,
                         self.running_mean, self.running_var)
        return y


class _BatchNorm2d(Operation):

    def __init__(self, handle, running_mean, running_var):
        self.running_mean = running_mean.data
        self.running_var = running_var.data
        self.handle = handle

    def forward(self, x, scale, bias):
        if training:

            if self.handle.device_id == -1:
                raise NotImplementedError
            else:
                y, mean, var = singa.GpuBatchNormForwardTraining(self.handle,
                                                                 x, scale, bias, self.running_mean, self.running_var)
                self.cache = (x, scale, mean, var)
        else:
            if self.handle.device_id == -1:
                raise NotImplementedError
            else:
                y = singa.GpuBatchNormForwardInference(
                    self.handle, x, scale, bias, self.running_mean, self.running_var)
        return y

    def backward(self, dy):
        assert training is True and hasattr(
            self, 'cache'), 'Please set training as True before do BP. '

        if dy.device().id() != self.handle.device_id:
            dy.ToDevice(self.cache[0].device())

        if self.handle.device_id == -1:
            raise NotImplementedError
        else:
            x, scale, mean, var = self.cache
            dx, ds, db = singa.GpuBatchNormBackward(
                self.handle, dy, x, scale, mean, var)
            return dx, ds, db


def batchnorm_2d(handle, x, scale, bias, running_mean, running_var):
    return _BatchNorm2d(handle, running_mean, running_var)(x, scale, bias)[0]


class _Pooling2d(Operation):

    def __init__(self, handle):
        self.handle = handle

    def forward(self, x):
        if self.handle.device_id == -1:
            raise NotImplementedError
        else:
            y = singa.GpuPoolingForward(self.handle, x)

        if training:
            self.cache = (x, y)

        return y

    def backward(self, dy):
        if self.handle.device_id == -1:
            raise NotImplementedError
        else:
            dx = singa.GpuPoolingBackward(self.handle,
                                          dy, self.cache[0], self.cache[1])
        return dx


def pooling_2d(handle, x):
    return _Pooling2d(handle)(x)[0]


class Pooling2d(Layer):

    def __init__(self, kernel_size, stride=None, padding=0, is_max=True):
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise TypeError('Wrong kernel_size type.')

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
            assert stride[0] > 0 or (kernel_size[0] == 1 and padding[
                0] == 0), 'stride[0]=0, but kernel_size[0]=%d, padding[0]=%d' % (kernel_size[0], padding[0])
        else:
            raise TypeError('Wrong stride type.')

        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple):
            self.padding = padding
        else:
            raise TypeError('Wrong padding type.')

        self.is_max = is_max

    def __call__(self, x):

        out_shape_h = int(
            (x.shape[2] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0]) + 1
        out_shape_w = int(
            (x.shape[3] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1]) + 1
        if x.device.id() == -1:
            if not hasattr(self, 'handle'):
                self.handle = singa.PoolingHandle(
                    x.data, self.kernel_size, self.stride, self.padding, self.is_max)
            elif x.shape[0] != self.handle.batchsize or out_shape_h != self.handle.pooled_height or \
                    out_shape_w != self.handle.pooled_width:
                self.handle = singa.PoolingHandle(x.data, self.kernel_size, self.stride,
                                                  self.padding, self.is_max)
        else:
            if not hasattr(self, 'handle'):
                self.handle = singa.CudnnPoolingHandle(x.data, self.kernel_size, self.stride,
                                                       self.padding, self.is_max)
            elif x.shape[0] != self.handle.batchsize or out_shape_h != self.handle.pooled_height or \
                    out_shape_w != self.handle.pooled_width:
                self.handle = singa.CudnnPoolingHandle(x.data, self.kernel_size, self.stride,
                                                       self.padding, self.is_max)

        self.handle.device_id = x.device.id()

        y = pooling_2d(self.handle, x)
        return y


class MaxPool2d(Pooling2d):

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPool2d, self).__init__(kernel_size, stride, padding, True)


class AvgPool2d(Pooling2d):

    def __init__(self, kernel_size, stride=None, padding=0):
        super(AvgPool2d, self).__init__(kernel_size, stride, padding, False)


class MaxPool1d(Pooling2d):

    def __init__(self, kernel_size, stride=None, padding=0):
        if stride is None:
            stride = kernel_size
        super(MaxPool2d, self).__init__(
            (1, kernel_size), (0, stride), (0, padding), True)


class AvgPool1d(Pooling2d):

    def __init__(self, kernel_size, stride=None, padding=0):
        if stride is None:
            stride = kernel_size
        super(MaxPool2d, self).__init__(
            (1, kernel_size), (0, stride), (0, padding), False)


class Tanh(Operation):

    def forward(self, x):
        out = singa.Tanh(x)
        if training:
            self.cache = (out,)
        return out

    def backward(self, dy):
        dx = singa.__mul__(self.cache[0], self.cache[0])
        dx = singa.MultFloat(dx, -1.0)
        dx = singa.AddFloat(dx, 1.0)
        dx = singa.__mul__(dy, dx)
        return dx


def tanh(x):
    return Tanh()(x)[0]


class Sigmoid(Operation):

    def forward(self, x):
        out = singa.Sigmoid(x)
        if training:
            self.cache = (out,)
        return out

    def backward(self, dy):
        dx = singa.MultFloat(self.cache[0], -1.0)
        dx = singa.AddFloat(dx, 1.0)
        dx = singa.__mul__(self.cache[0], dx)
        dx = singa.__mul__(dy, dx)
        return dx


def sigmoid(x):
    return Sigmoid()(x)[0]


class ElemMatmul(Operation):

    def forward(self, x1, x2):
        if training:
            self.cache = (x1, x2)
        return singa.__mul__(x1, x2)

    def backward(self, dy):
        dx1 = singa.__mul__(dy, self.cache[1])
        dx2 = singa.__mul__(dy, self.cache[0])
        return dx1, dx2


def mul(x, y):
    # do pointwise multiplication
    return ElemMatmul()(x, y)[0]


def add_all(*xs):
    assert len(xs) > 2
    y = add(xs[0], xs[1])
    for x in xs[2:]:
        y = add(y, x)
    return


class RNN_Base(Layer):

    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    def step_forward(self):
        raise NotImplementedError


class RNN(RNN_Base):

    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0, bidirectional=False):
        self.nonlinearity = nonlinearity

        Wx_shape = (input_size, hidden_size)
        self.Wx = Tensor(shape=Wx_shape, requires_grad=True, stores_grad=True)
        self.Wx.gaussian(0.0, 1.0)

        Wh_shape = (hidden_size, hidden_size)
        self.Wh = Tensor(shape=Wh_shape, requires_grad=True, stores_grad=True)
        self.Wh.gaussian(0.0, 1.0)

        B_shape = (hidden_size,)
        self.b = Tensor(shape=B_shape, requires_grad=True, stores_grad=True)
        self.b.set_value(0.0)

        self.params = (self.Wx, self.Wh, self.b)

    def __call__(self, xs, h0):
        # xs: a tuple or list of input tensors
        if not isinstance(xs, tuple):
            xs = tuple(xs)
        inputs = xs + (h0,)
        self.device_check(*inputs)
        #self.device_check(inputs[0], *self.params)
        self.device_check(inputs[0], self.Wx, self.Wh, self.b)
        batchsize = xs[0].shape[0]
        out = []
        h = self.step_forward(xs[0], h0, self.Wx, self.Wh, self.b)
        out.append(h)
        for x in xs[1:]:
            assert x.shape[0] == batchsize
            h = self.step_forward(x, h, self.Wx, self.Wh, self.b)
            out.append(h)
        return out, h

    def step_forward(self, x, h, Wx, Wh, b):
        y2 = matmul(h, Wh)
        y1 = matmul(x, Wx)
        y = add(y2, y1)
        y = add_bias(y, b, axis=0)
        if self.nonlinearity == 'tanh':
            y = tanh(y)
        elif self.nonlinearity == 'relu':
            y = relu(y)
        else:
            raise ValueError
        return y


class LSTM(RNN_Base):

    def __init__(self, input_size, hidden_size, nonlinearity='tanh', num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False):
        self.nonlinearity = nonlinearity

        Wx_shape = (input_size, hidden_size)
        self.Wx = []
        for i in range(4):
            w = Tensor(shape=Wx_shape, requires_grad=True, stores_grad=True)
            w.gaussian(0.0, 1.0)
            self.Wx.append(w)

        Wh_shape = (hidden_size, hidden_size)
        self.Wh = []
        for i in range(4):
            w = Tensor(shape=Wh_shape, requires_grad=True, stores_grad=True)
            w.gaussian(0.0, 1.0)
            self.Wh.append(w)

        Bx_shape = (hidden_size,)
        self.Bx = []
        for i in range(4):
            b = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
            b.set_value(0.0)
            self.Bx.append(b)

        Bh_shape = (hidden_size,)
        self.Bh = []
        for i in range(4):
            b = Tensor(shape=Bx_shape, requires_grad=True, stores_grad=True)
            b.set_value(0.0)
            self.Bh.append(b)

        self.params = self.Wx + self.Wh + self.Bx + self.Bh

    def __call__(self, xs, (h0, c0)):
        # xs: a tuple or list of input tensors
        if not isinstance(xs, list):
            xs = list(xs)
        inputs = xs + list((h0, c0))
        self.device_check(*inputs)
        #self.device_check(inputs[0], *self.params)
        self.device_check(inputs[0], *(self.Wx + self.Wh + self.Bx + self.Bh))
        batchsize = xs[0].shape[0]
        out = []
        h, c = self.step_forward(
            xs[0], h0, c0, self.Wx, self.Wh, self.Bx, self.Bh)
        out.append(h)
        for x in xs[1:]:
            assert x.shape[0] == batchsize
            h, c = self.step_forward(
                x, h, c, self.Wx, self.Wh, self.Bx, self.Bh)
            out.append(h)
        return out, h, c

    def step_forward(self, x, h, c, Wx, Wh, Bx, Bh):
        y1 = matmul(x, Wx[0])
        y1 = add_bias(y1, Bx[0], axis=0)
        y2 = matmul(h, Wh[0])
        y2 = add_bias(y2, Bh[0], axis=0)
        i = add(y1, y2)
        i = sigmoid(i)

        y1 = matmul(x, Wx[1])
        y1 = add_bias(y1, Bx[1], axis=0)
        y2 = matmul(h, Wh[1])
        y2 = add_bias(y2, Bh[1], axis=0)
        f = add(y1, y2)
        f = sigmoid(f)

        y1 = matmul(x, Wx[2])
        y1 = add_bias(y1, Bx[2], axis=0)
        y2 = matmul(h, Wh[2])
        y2 = add_bias(y2, Bh[2], axis=0)
        o = add(y1, y2)
        o = sigmoid(o)

        y1 = matmul(x, Wx[3])
        y1 = add_bias(y1, Bx[3], axis=0)
        y2 = matmul(h, Wh[3])
        y2 = add_bias(y2, Bh[3], axis=0)
        g = add(y1, y2)
        g = tanh(g)

        cout1 = mul(f, c)
        cout2 = mul(i, g)
        cout = add(cout1, cout2)

        hout = tanh(cout)
        hout = mul(o, hout)
        return hout, cout
