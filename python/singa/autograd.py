
from __future__ import division

from functools import reduce
from collections import Counter, deque
from .tensor import Tensor, Dummy, Operation

from singa import layer
from singa.proto import model_pb2
from . import singa_wrap as singa

import numpy as np

CTensor = singa.Tensor


class ReLU(Operation):

    def forward(self, x):
        '''
        Args:
            x(CTensor): input tensor

        Returns:
            a new CTensor whose element y = x if x >= 0; otherwise 0;
        '''
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
        self.input = (x, w)
        return singa.Mult(x, w)

    def backward(self, dy):
        '''
        Args:
            dy (CTensor): data for the dL / dy, L is the loss

        Returns:
            a tuple for (dx, dw)
        '''
        return singa.Mult(dy, self.input[1].T()), \
            singa.Mult(self.input[0].T(), dy)


def matmul(x, w):
    return Matmul()(x, w)[0]


class AddBias(Operation):
    '''
    Add Bias to each row / column of the Tensor, depending on the parameter axis.
    '''

    def __init__(self, axis=0):
        '''
        To indicate the calculation axis, 0 for row, 1 for column.

        Args:
            axis: 0 or 1, default is 0.
        '''
        self.axis = axis

    def forward(self, x, b):
        '''
        Args:
            x: matrix.
            b: bias to be added.

        Return:
            the result Tensor
        '''
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


class SoftMax(Operation):
    '''
    Apply SoftMax for each row of the Tensor or each column of the Tensor
    according to the parameter axis.
    '''

    def __init__(self, axis=0):
        self.axis = axis

    def forward(self, x):
        '''
        Args:
            x(data): the input 1d or 2d tensor

        Returns:
            the result Tensor
        '''
        if self.axis == 1:
            x = x.T()
        self.output = singa.SoftMax(x)
        if self.axis == 0:
            return self.output
        elif self.axis == 1:
            return self.output.T()

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
            dy = dy.T()
        grad = ctensor2numpy(dy)
        output = ctensor2numpy(self.output)
        out_1 = np.einsum('ki,ki->ki', grad, output)
        medium_out = np.einsum('ki,kj->kij', output, output)
        out_2 = np.einsum('kij,kj->ki', medium_out, grad)
        out = out_1 - out_2
        dx = CTensor(out_1.shape)
        dx.CopyFloatDataFromHostPtr(out.flatten())
        if self.axis == 0:
            return dx
        elif self.axis == 1:
            return dx.T()


def soft_max(x, axis=0):
    return SoftMax(axis)(x)[0]


class CrossEntropy(Operation):
    '''
    Calculte CrossEntropy loss for a batch of training data.

    '''

    def forward(self, x, t):
        '''
        Args:
            x (CTensor): 1d or 2d tensor, the prediction data(output) of current network.
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
            dy (float or CTensor): scalar, accumulate gradient from outside of current network, usually
            equal to 1.0

        Returns:
            dx (CTensor): data for the dL /dx, L is the loss, x is the output of current network.
            note that this is true for dy = 1.0
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


def ctensor2numpy(x):
    '''
    To be used in SoftMax Operation.
    Convert a singa_tensor to numpy_tensor.
    '''
    np_array = x.GetFloatValue(int(x.Size()))
    return np_array.reshape(x.shape())

class Conv2d(Operation):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):

        name = 'Conv2d'
        border_mode = 'same'
        cudnn_prefer = 'fastest'
        workspace_byte_limit = 1024
        data_format = 'NCHW'
        W_specs ={'init': 'gaussian',
                  'mean':0.0,
                  'std':0.1}
        b_specs = {'init': 'constant'}
        input_sample_shape = None

        inner_params = {'name':name,
                          'border_mode':border_mode,
                          'cudnn_prefer':cudnn_prefer,
                          'workspace_byte_limit':workspace_byte_limit,
                          'data_format':data_format,
                          'W_specs':W_specs,
                          'b_specs':b_specs,
                          'input_sample_shape':input_sample_shape
                          }
        # TODO valid value of inner_params check

        for kwarg in kwargs:
            if kwarg not in inner_params:
                raise TypeError('Keyword argument not understood:', kwarg)
            else:
                inner_params[kwarg] = kwargs[kwarg]
                

        self.W_specs=inner_params['W_specs']
        self.b_specs=inner_params['b_specs']

        if padding == 0:
            pad = None
        else:
            pad = padding

        if dilation != 1 or groups != 1:
            raise ValueError('Not implemented yet')

        self.PyLayer = layer.Conv2D(inner_params['name'], nb_kernels=out_channels, kernel=kernel_size, stride=stride,
                                    border_mode=inner_params['border_mode'],
                 cudnn_prefer=inner_params['cudnn_prefer'], workspace_byte_limit=inner_params['workspace_byte_limit'],
                 data_format=inner_params['data_format'], use_bias=bias, W_specs=self.W_specs, b_specs=self.b_specs,
                 pad=pad, input_sample_shape=inner_params['input_sample_shape'])

    def __call__(self, x, flag=True):
        assert type(flag) is bool, 'flag can only be bool.'
        if flag:
            self.flag = model_pb2.kTrain
        else:
            self.flag = model_pb2.kEval

        if not self.PyLayer.has_setup:
            self.PyLayer.setup(x.shape[1:])

        param_data = self.PyLayer.layer.param_values()
        if not hasattr(self, 'w'):
            self.w = Tensor(data=param_data[0], requires_grad=True, stores_grad=True)
            if self.W_specs['init'] == 'gaussian':
                if 'std' not in self.W_specs or 'mean' not in self.W_specs:
                    self.w.gaussian(0.0, 0.1)
                else:
                    self.w.gaussian(self.W_specs['mean'],self.W_specs['std'])
            elif self.W_specs['init'] == 'uniform':
                if 'low' not in self.W_specs or 'high' not in self.W_specs:
                    self.w.uniform(0.0, 1.0)
                else:
                    self.w.uniform(self.W_specs['low'],self.W_specs['high'])
            elif self.W_specs['init'] == 'xavier':
                pass  # TODO

        xs = [x, self.w]

        if len(param_data) == 2:
            if not hasattr(self, 'b'):
                self.b = Tensor(data=param_data[1], requires_grad=True, stores_grad=True)
                if self.b_specs['init'] == 'gaussian':
                    if 'std' not in self.b_specs or 'mean' not in self.b_specs:
                        self.b.gaussian(0.0, 0.1)
                    else:
                        self.b.gaussian(self.b_specs['mean'], self.b_specs['std'])
                elif self.b_specs['init'] == 'uniform':
                    if 'low' not in self.b_specs or 'high' not in self.b_specs:
                        self.b.uniform(0.0, 1.0)
                    else:
                        self.b.uniform(self.b_specs['low'], self.b_specs['high'])
                elif self.b_specs['init'] == 'xavier':
                    pass  # TODO
                elif self.b_specs['init'] == 'constant':
                    self.b.set_value(0.0)

            xs.append(self.b)

        xs = tuple(xs)
        return self._do_forward(*xs)

    def forward(self, *xs):
        return self.PyLayer.layer.Forward(self.flag, xs[0])

    def backward(self, dy):
        ret = self.PyLayer.layer.Backward(0, dy)
        return (ret[0],)+ret[1]


class MaxPool2d(Operation):
    def __init__(self, kernel_size=3, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False, **kwargs):

        name = 'MaxPool2d'
        border_mode = 'same'
        data_format = 'NCHW'
        input_sample_shape = None

        allowed_kwargs = {'name': name,
                          'border_mode': border_mode,
                          'data_format': data_format,
                          'input_sample_shape': input_sample_shape
                          }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)
            else:
                allowed_kwargs[kwarg] = kwargs[kwarg]

        if padding == 0:
            pad = None
        else:
            pad = padding

        if dilation != 1 or return_indices is not False or ceil_mode is not False:
            raise ValueError('Not implemented yet')

        self.PyLayer = layer.Pooling2D(name, model_pb2.PoolingConf.MAX,
                                           kernel_size, stride, border_mode,
                                           pad, data_format, input_sample_shape)

    def __call__(self, x, flag=True):
        assert type(flag) is bool, 'flag can only be bool.'
        if flag:
            self.flag = model_pb2.kTrain
        else:
            self.flag = model_pb2.kEval

        if not self.PyLayer.has_setup:
            self.PyLayer.setup(x.shape[1:])

        return self._do_forward(x)

    def forward(self, *xs):
        return self.PyLayer.layer.Forward(self.flag, xs[0])

    def backward(self, dy):
        return self.PyLayer.layer.Backward(0, dy)[0]


class ReLU_Layer(Operation):
    def __init__(self, name='ReLU', mode='relu',input_sample_shape=None):
        self.PyLayer = layer.Activation(name, mode, input_sample_shape)

    def __call__(self, x, flag=True):
        assert type(flag) is bool, 'flag can only be bool.'
        if flag:
            self.flag = model_pb2.kTrain
        else:
            self.flag = model_pb2.kEval
        if not self.PyLayer.has_setup:
            self.PyLayer.setup(x.shape[1:])
        return self._do_forward(x)

    def forward(self, *xs):
        return self.PyLayer.layer.Forward(self.flag, xs[0])

    def backward(self, dy):
        return self.PyLayer.layer.Backward(0, dy)[0]


class Flatten(Operation):
    def __init__(self, name='Flatten', axis=1, input_sample_shape=None):
        self.PyLayer = layer.Flatten(name, axis, input_sample_shape)

    def __call__(self, x, flag=True):
        assert type(flag) is bool, 'flag can only be bool.'
        if flag:
            self.flag = model_pb2.kTrain
        else:
            self.flag = model_pb2.kEval
        if not self.PyLayer.has_setup:
            self.PyLayer.setup(x.shape[1:])
        return self._do_forward(x)

    def forward(self, *xs):
        return self.PyLayer.layer.Forward(self.flag, xs[0])

    def backward(self, dy):
        return self.PyLayer.layer.Backward(0, dy)[0]


class Linear(Operation):
    def __init__(self, in_features, out_features, bias=True, **kwargs):

        name = 'Linear'
        W_transpose=False
        W_specs = {'init': 'gaussian',
                   'mean': 0.0,
                   'std': 0.1}
        b_specs = {'init': 'constant'}
        input_sample_shape = in_features

        inner_params = {'name': name,
                          'W_transpose': W_transpose,
                          'W_specs': W_specs,
                          'b_specs': b_specs,
                          'input_sample_shape': input_sample_shape
                          }

        # TODO valid value of inner_params check

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)
            else:
                inner_params[kwarg] = kwargs[kwarg]

        self.W_specs = W_specs
        self.b_specs = b_specs

        self.PyLayer = layer.Dense(inner_params['name'], num_output=out_features, use_bias=bias,
                     W_specs=self.W_specs, b_specs=self.b_specs,
                     W_transpose=inner_params['W_transpose'], input_sample_shape=inner_params['input_sample_shape'])

    def __call__(self, x, flag=True):
        assert type(flag) is bool, 'flag can only be bool.'
        if flag:
            self.flag = model_pb2.kTrain
        else:
            self.flag = model_pb2.kEval

        if not self.PyLayer.has_setup:
            self.PyLayer.setup(x.shape[1:])

        param_data = self.PyLayer.layer.param_values()
        if not hasattr(self, 'w'):
            self.w = Tensor(data=param_data[0], requires_grad=True, stores_grad=True)
            if self.W_specs['init'] == 'gaussian':
                if 'std' not in self.W_specs or 'mean' not in self.W_specs:
                    self.w.gaussian(0.0, 0.1)
                else:
                    self.w.gaussian(self.W_specs['mean'],self.W_specs['std'])
            elif self.W_specs['init'] == 'uniform':
                if 'low' not in self.W_specs or 'high' not in self.W_specs:
                    self.w.uniform(0.0, 1.0)
                else:
                    self.w.uniform(self.W_specs['low'],self.W_specs['high'])
            elif self.W_specs['init'] == 'xavier':
                pass  # TODO

        xs = [x, self.w]

        if len(param_data) == 2:
            if not hasattr(self, 'b'):
                self.b = Tensor(data=param_data[1], requires_grad=True, stores_grad=True)
                if self.b_specs['init'] == 'gaussian':
                    if 'std' not in self.b_specs or 'mean' not in self.b_specs:
                        self.b.gaussian(0.0, 0.1)
                    else:
                        self.b.gaussian(self.b_specs['mean'], self.b_specs['std'])
                elif self.b_specs['init'] == 'uniform':
                    if 'low' not in self.b_specs or 'high' not in self.b_specs:
                        self.b.uniform(0.0, 1.0)
                    else:
                        self.b.uniform(self.b_specs['low'], self.b_specs['high'])
                elif self.b_specs['init'] == 'xavier':
                    pass  # TODO
                elif self.b_specs['init'] == 'constant':
                    self.b.set_value(0.0)

            xs.append(self.b)

        xs = tuple(xs)
        return self._do_forward(*xs)

    def forward(self, *xs):
        return self.PyLayer.layer.Forward(self.flag, xs[0])

    def backward(self, dy):
        ret = self.PyLayer.layer.Backward(0, dy)
        return (ret[0],)+ret[1]

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
    # dependency = {}
    dependency_count = Counter()
    queue = deque([op])
    while len(queue) > 0:
        cur_op = queue.pop()
        for src_op, _, _, _ in cur_op.src:
            if src_op not in dependency_count and \
                    (not isinstance(src_op, Dummy)):
                # dependency[src_op] = [Counter() for _ in src_op.y_id2idx]
                dependency_count[src_op] = 0
                queue.append(src_op)
            # y_idx = src_op.y_id2idx[x_id]
            # dependency[src_op][y_idx][cur_op] += 1
            dependency_count[src_op] += 1
    return dependency_count


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
    gradients = {}  # mapping: x->dx if x.stores_grad
    if y.stores_grad:
        gradients[y] = dy

    while len(ready) > 0:
        op, dys = ready.pop()
        if not op.requires_grad or isinstance(op, Dummy):
            continue
        # if not isinstance(op, tensor.Dummy):
        dxs = op._do_backward(*dys)
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
            if y_stores_grad:
                # store the gradient for final return, e.g. if x is parameter
                g = not_ready[src_op][y_idx]
                gradients[y] = Tensor(device=g.device, data=g)
            dependency[src_op] -= 1
            if src_op.requires_grad is True:
                if dependency[src_op] == 0:
                    if not isinstance(src_op, Dummy):
                        ready.append((src_op, not_ready[src_op]))
                    del not_ready[src_op]

    return gradients
