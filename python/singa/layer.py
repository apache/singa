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

import math
from functools import wraps
from collections import OrderedDict

from singa import utils
from .tensor import Tensor
from . import singa_wrap as singa


class LayerMeta(type):

    def init_wrapper(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if len(args) == 0:
                return

            if isinstance(args[0], list):
                assert len(args) > 0 and isinstance(args[0][0], Tensor), (
                    'initialize function expects PlaceHolders or Tensors')
                dev = args[0][0].device
            else:
                assert len(args) > 0 and isinstance(args[0], Tensor), (
                    'initialize function expects PlaceHolders or Tensors')
                dev = args[0].device

            self._get_unique_name()
            prev_state = dev.graph_enabled()
            dev.EnableGraph(False)
            func(self, *args, **kwargs)
            self._initialized = True
            dev.EnableGraph(prev_state)

        return wrapper

    def forward_wrapper(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self._initialized:
                self.initialize(*args, **kwargs)
                self._initialized = True
            return func(self, *args, **kwargs)

        return wrapper

    def __new__(cls, name, bases, attr):
        if 'initialize' in attr:
            attr['initialize'] = LayerMeta.init_wrapper(attr['initialize'])
        if 'forward' in attr:
            attr['forward'] = LayerMeta.forward_wrapper(attr['forward'])

        return super(LayerMeta, cls).__new__(cls, name, bases, attr)


class Layer(object, metaclass=LayerMeta):

    sep = '.'

    def __init__(self):
        self.name = self.__class__.__name__
        self._initialized = False
        self._parent = None
        self._layers = dict()

    def initialize(self, *input):
        pass

    def forward(self, *input):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_params(self):
        params = dict()
        sublayers = self._layers
        for name, sublayer in sublayers.items():
            if sublayer._initialized:
                params.update(sublayer.get_params())
        return params

    def set_params(self, parameters):
        # set parameters for Layer
        # input should be either a PyTensor or numpy ndarray.
        # examples: Layer.set_params(W=np.ones((in, out), dtype=np.float32)),
        # Layer.set_params(**{'block1':{'linear1':{'W':np.ones((in, out),
        # dtype=np.float32)}}})
        names = parameters.keys()
        sublayers = self._layers
        for name, sublayer in sublayers.items():
            if sublayer._initialized:
                if self.has_layer_param(sublayer, names):
                    sublayer.set_params(parameters)

    def get_states(self):
        states = dict()
        sublayers = self._layers
        for name, sublayer in sublayers.items():
            if sublayer._initialized:
                states.update(sublayer.get_states())
        states.update(self.get_params())
        return states

    def set_states(self, states):
        names = states.keys()
        sublayers = self._layers
        for name, sublayer in sublayers.items():
            if sublayer._initialized:
                if self.has_layer_param(sublayer, names):
                    sublayer.set_states(states)
        self.set_params(states)

    def has_layer_param(self, layer, names):
        for name in names:
            if name.startswith(layer.name):
                return True
        return False

    def device_check(self, *inputs):
        # disabled the graph to prevent buffering data transfer operator
        x_device = inputs[0].device
        prev_state = x_device.graph_enabled()
        x_device.EnableGraph(False)
        x_dev_id = x_device.id()
        for var in inputs:
            if var.device.id() != x_dev_id:
                var.to_device(x_device)
        x_device.EnableGraph(prev_state)

    def _get_unique_name(self):
        prefix = ''
        if self._parent:
            prefix = self._parent.name
            if prefix:
                prefix += Layer.sep
            self.name = prefix + self.name
        else:
            self.name = ''
        return self.name

    def __getattr__(self, name):
        if '_layers' in self.__dict__:
            layers = self.__dict__['_layers']
            if name in layers:
                return layers[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        if isinstance(value, Layer):
            # TODO: remove the attr from dict first
            self.__dict__['_layers'][name] = value
            value.__dict__['_parent'] = self
            value.__dict__['name'] = name
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._layers:
            del self._layers[name]
        else:
            object.__delattr__(self, name)

    def register_layers(self, *args):
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            items = args[0].items()
        else:
            items = [(v.__class__.__name__ + '_' + str(idx), v)
                     for idx, v in enumerate(args)]

        for name, value in items:
            if isinstance(value, Layer):
                self._layers[name] = value
                value.__dict__['_parent'] = self
                value.name = name


class Linear(Layer):
    """
    Generate a Linear operator
    """

    # TODO: replace current with
    #   def __init__(self, out_features, bias=True):
    def __init__(self, out_features, *args, bias=True, **kwargs):
        """
        Args:
            out_channels: int, the channel of output, also is the number of
                filters
            bias: bool
        """
        super(Linear, self).__init__()

        self.out_features = out_features

        # TODO: for backward compatibility, to remove
        if len(args) > 0:
            self.in_features = out_features
            self.out_features = args[0]
        if len(args) > 1:
            self.bias = args[1]
        else:
            self.bias = bias

    def initialize(self, x):
        self.in_features = x.shape[1]
        w_shape = (self.in_features, self.out_features)
        b_shape = (self.out_features,)
        w_name = self.name + Layer.sep + 'W'
        b_name = self.name + Layer.sep + 'b'

        self.W = Tensor(shape=w_shape,
                        requires_grad=True,
                        stores_grad=True,
                        name=w_name)
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        self.W.gaussian(0.0, std)

        if self.bias:
            self.b = Tensor(shape=b_shape,
                            requires_grad=True,
                            stores_grad=True,
                            name=b_name)
            self.b.set_value(0.0)
        else:
            self.b = None

    def forward(self, x):
        if self.b:
            self.device_check(x, self.W, self.b)
        else:
            self.device_check(x, self.W)

        assert x.shape[1] == self.W.shape[0], (
            "Linear layer expects input features size %d received %d" %
            (self.W.shape[0], x.shape[1]))

        y = autograd.matmul(x, self.W)
        if self.bias:
            y = autograd.add_bias(y, self.b, axis=0)
        return y

    def get_params(self):
        if self.bias:
            return {self.W.name: self.W, self.b.name: self.b}
        else:
            return {self.W.name: self.W}

    def set_params(self, parameters):
        self.W.copy_from(parameters[self.W.name])
        if self.bias:
            self.b.copy_from(parameters[self.b.name])


class Gemm(Layer):
    """
    Generate a Gemm operator
    Y = alpha * A' * B' + beta * C
    B is weight, C is bias
    """

    def __init__(self,
                 nb_kernels,
                 alpha=1.0,
                 beta=1.0,
                 transA=False,
                 transB=True,
                 bias=True,
                 bias_shape=None):
        """
        Args:
            nb_kernels: int, the channel of output, also is the number of
                filters
            alpha (float): Scalar multiplier for the product of input tensors A * B.
            beta (float): Scalar multiplier for input tensor C.
            ransA (bool): Whether A should be transposed
            transB (bool): Whether B should be transposed
            bias: bool
        """
        super(Gemm, self).__init__()
        self.nb_kernels = nb_kernels
        self.alpha = alpha
        self.beta = beta
        self.transA = 1 if transA else 0
        self.transB = 1 if transB else 0
        self.bias = bias
        self.bias_shape = bias_shape

    def initialize(self, x):
        if self.transA == 0:
            self.in_features = x.shape[1]
        else:
            self.in_features = x.shape[0]

        if self.transB == 0:
            w_shape = (self.in_features, self.nb_kernels)
        else:
            w_shape = (self.nb_kernels, self.in_features)

        if self.bias_shape:
            b_shape = self.bias_shape
        else:
            b_shape = (1, self.nb_kernels)

        w_name = self.name + Layer.sep + 'W'
        b_name = self.name + Layer.sep + 'b'

        self.W = Tensor(shape=w_shape,
                        requires_grad=True,
                        stores_grad=True,
                        device=x.device,
                        name=w_name)
        std = math.sqrt(2.0 / (self.in_features + self.nb_kernels))
        self.W.gaussian(0.0, std)

        if self.bias:
            self.b = Tensor(shape=b_shape,
                            requires_grad=True,
                            stores_grad=True,
                            device=x.device,
                            name=b_name)
            self.b.set_value(0.0)
        else:
            self.b = None

    def forward(self, x):
        if self.b:
            self.device_check(x, self.W, self.b)
        else:
            self.device_check(x, self.W)

        if self.transA == 0:
            in_features = x.shape[1]
        else:
            in_features = x.shape[0]

        if self.transB == 0:
            in_features_w = self.W.shape[0]
        else:
            in_features_w = self.W.shape[1]

        assert in_features == in_features_w, (
            "Gemm layer expects input features size %d received %d" %
            (in_features_w, in_features))
        y = autograd.gemm(x, self.W, self.b, self.alpha, self.beta, self.transA,
                          self.transB)

        return y

    def get_params(self):
        if self.bias:
            return {self.W.name: self.W, self.b.name: self.b}
        else:
            return {self.W.name: self.W}

    def set_params(self, parameters):
        self.W.copy_from(parameters[self.W.name])
        if self.bias:
            self.b.copy_from(parameters[self.b.name])


class Conv2d(Layer):
    """
    Generate a Conv 2d operator
    """

    def __init__(self,
                 nb_kernels,
                 kernel_size,
                 *args,
                 stride=1,
                 padding=0,
                 dilation=1,
                 group=1,
                 bias=True,
                 pad_mode="NOTSET",
                 activation="NOTSET",
                 **kwargs):
        """
        Args:
            nb_kernels (int): the channel of output, also is the number of filters
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            dilation (int): only support 1
            group (int): group
            bias (bool): bias
            pad_mode (string): can be NOTSET, SAME_UPPER, or SAME_LOWER, where
                default value is NOTSET, which means explicit padding is used.
                SAME_UPPER or SAME_LOWER mean pad the input so that the output
                spatial size match the input. In case of odd number add the extra
                padding at the end for SAME_UPPER and at the beginning for SAME_LOWER.
            activation (string): can be NOTSET, RELU, where default value is NOTSET,
                which means there is no activation behind the conv2d layer.
                RELU means there is a ReLU behind current conv2d layer.
        """
        super(Conv2d, self).__init__()

        # the old code create the layer like: Conv2d(8, 16, 3)， or Conv2d(8, 16, 3, stride=1)
        # the following code block is for backward compatibility
        if len(args) > 0:
            nb_kernels = kernel_size
            kernel_size = args[0]
        if len(args) > 1:
            stride = args[1]
        if len(args) > 2:
            padding = args[2]

        self.nb_kernels = nb_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.group = group
        self.bias = bias
        self.pad_mode = pad_mode
        self.activation = activation

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise TypeError("Wrong kernel_size type.")

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
        else:
            raise TypeError("Wrong stride type.")

        self.odd_padding = (0, 0, 0, 0)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple) or isinstance(padding, list):
            if len(padding) == 2:
                self.padding = padding
            elif len(padding) == 4:
                _h_mask = padding[0] - padding[1]
                _w_mask = padding[2] - padding[3]
                # the odd paddding is the value that cannot be handled by the tuple padding (w, h) mode
                # so we need to firstly handle the input, then use the nomal padding method.
                self.odd_padding = (max(_h_mask, 0), max(-_h_mask, 0),
                                    max(_w_mask, 0), max(-_w_mask, 0))
                self.padding = (
                    padding[0] - self.odd_padding[0],
                    padding[2] - self.odd_padding[2],
                )
            else:
                raise TypeError("Wrong padding value.")

        if dilation != 1 and list(dilation) != [1, 1]:
            raise ValueError("Not implemented yet")

        self.inner_params = {
            "cudnn_prefer": "fastest",
            "workspace_MB_limit": 1024,
        }
        # TODO valid value of inner_params check

        for kwarg in kwargs:
            if kwarg not in self.inner_params:
                raise TypeError("Keyword argument not understood:", kwarg)
            else:
                self.inner_params[kwarg] = kwargs[kwarg]

    def initialize(self, x):
        self.in_channels = x.shape[1]
        w_shape = (
            self.nb_kernels,
            int(self.in_channels / self.group),
            self.kernel_size[0],
            self.kernel_size[1],
        )
        w_name = self.name + Layer.sep + 'W'

        self.W = Tensor(shape=w_shape,
                        requires_grad=True,
                        stores_grad=True,
                        name=w_name,
                        device=x.device)
        # std = math.sqrt(
        # 2.0 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1] +
        # self.nb_kernels))
        std = math.sqrt(
            2.0 / (w_shape[1] * self.kernel_size[0] * self.kernel_size[1] +
                   self.nb_kernels))
        self.W.gaussian(0.0, std)

        if self.bias:
            b_shape = (self.nb_kernels,)
            b_name = self.name + Layer.sep + 'b'
            self.b = Tensor(shape=b_shape,
                            requires_grad=True,
                            stores_grad=True,
                            name=b_name,
                            device=x.device)
            self.b.set_value(0.0)
        else:
            # to keep consistency when to do forward.
            self.b = None
            # Tensor(data=CTensor([]), requires_grad=False, stores_grad=False)

        # if same pad mode, re-compute the padding
        if self.pad_mode in ("SAME_UPPER", "SAME_LOWER"):
            self.padding, self.odd_padding = utils.get_padding_shape(
                self.pad_mode, x.shape[2:], self.kernel_size, self.stride)
            self.padding = [self.padding[0], self.padding[2]]

        _x = x
        if self.odd_padding != (0, 0, 0, 0):
            x_shape = list(x.data.shape())
            x_shape[2] += (self.odd_padding[0] + self.odd_padding[1])
            x_shape[3] += (self.odd_padding[2] + self.odd_padding[3])
            _x = Tensor(shape=x_shape, device=x.device)
            _x.set_value(0.0)

        if _x.device.id() == -1:
            if self.group != 1:
                raise ValueError("Not implemented yet")
            else:
                if not hasattr(self, "handle"):
                    self.handle = singa.ConvHandle(
                        _x.data,
                        self.kernel_size,
                        self.stride,
                        self.padding,
                        self.in_channels,
                        self.nb_kernels,
                        self.bias,
                        self.group,
                    )
        else:
            if not hasattr(self, "handle"):
                self.handle = singa.CudnnConvHandle(
                    _x.data,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.in_channels,
                    self.nb_kernels,
                    self.bias,
                    self.group,
                )

    def forward(self, x):
        # sanitize the device of params/states, TODO: better to decorate forward()
        self.device_check(x, *[s for k, s in self.get_states().items()])

        assert (self.group >= 1 and self.in_channels %
                self.group == 0), "please set reasonable group."

        assert (self.nb_kernels >= self.group and self.nb_kernels %
                self.group == 0), "nb_kernels and group dismatched."

        y = autograd.conv2d(self.handle, x, self.W, self.b, self.odd_padding)

        if self.activation != "NOTSET":
            if self.activation == "RELU":
                y = autograd.relu(y)

        return y

    def get_params(self):
        if self.bias:
            return {self.W.name: self.W, self.b.name: self.b}
        else:
            return {self.W.name: self.W}

    def set_params(self, parameters):
        self.W.copy_from(parameters[self.W.name])
        if self.bias:
            self.b.copy_from(parameters[self.b.name])


class SeparableConv2d(Layer):
    """
    Generate a Conv 2d operator
    """

    def __init__(self,
                 nb_kernels,
                 kernel_size,
                 *args,
                 stride=1,
                 padding=0,
                 bias=False):
        """
        Args:
            nb_kernels (int): the channel of output, also is the number of filters
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            bias (bool): bias
        """
        super(SeparableConv2d, self).__init__()

        # the following code block is for backward compatibility
        if len(args) > 0:
            nb_kernels = kernel_size
            kernel_size = args[0]
        if len(args) > 1:
            stride = args[1]
        if len(args) > 2:
            padding = args[2]

        self.nb_kernels = nb_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

    def initialize(self, x):
        self.in_channels = x.shape[1]
        self.depthwise_conv = Conv2d(
            self.in_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            group=self.in_channels,
            bias=self.bias,
        )

        self.point_conv = Conv2d(self.nb_kernels, 1, bias=self.bias)

    def forward(self, x):
        y = self.depthwise_conv(x)
        y = self.point_conv(y)
        return y


class BatchNorm2d(Layer):
    """
    Generate a BatchNorm 2d operator
    """

    def __init__(self, *args, momentum=0.9):
        """
        Args:
            momentum (float): Factor used in computing the running mean and
                variance.
        """
        super(BatchNorm2d, self).__init__()

        if len(args) > 0:
            self.channels = args[0]
        if len(args) > 1:
            self.momentum = args[1]
        self.momentum = momentum
        assert 0 <= momentum <= 1.0, ("Illegal momentum")

    def initialize(self, x):
        self.channels = x.shape[1]
        param_shape = (self.channels,)
        scale_name = self.name + Layer.sep + 'scale'
        bias_name = self.name + Layer.sep + 'bias'
        running_mean_name = self.name + Layer.sep + 'running_mean'
        running_var_name = self.name + Layer.sep + 'running_var'

        self.scale = Tensor(shape=param_shape,
                            requires_grad=True,
                            stores_grad=True,
                            name=scale_name)
        self.scale.set_value(1.0)

        self.bias = Tensor(shape=param_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=bias_name)
        self.bias.set_value(0.0)

        self.running_mean = Tensor(shape=param_shape,
                                   requires_grad=False,
                                   stores_grad=False,
                                   name=running_mean_name)
        self.running_mean.set_value(0.0)

        self.running_var = Tensor(shape=param_shape,
                                  requires_grad=False,
                                  stores_grad=False,
                                  name=running_var_name)
        self.running_var.set_value(1.0)

        if not hasattr(self, "handle"):
            if x.device.id() == -1:
                self.handle = singa.BatchNormHandle(self.momentum, x.data)
            else:
                self.handle = singa.CudnnBatchNormHandle(self.momentum, x.data)

    def forward(self, x):
        assert x.shape[1] == self.channels, (
            "number of channels dismatched. %d vs %d" %
            (x.shape[1], self.channels))

        self.device_check(x, self.scale, self.bias, self.running_mean,
                          self.running_var)

        y = autograd.batchnorm_2d(
            self.handle,
            x,
            self.scale,
            self.bias,
            self.running_mean,
            self.running_var,
        )
        return y

    def get_params(self):
        return {self.scale.name: self.scale, self.bias.name: self.bias}

    def set_params(self, parameters):
        self.scale.copy_from(parameters[self.scale.name])
        self.bias.copy_from(parameters[self.bias.name])

    def get_states(self):
        ret = self.get_params()
        ret[self.running_mean.name] = self.running_mean
        ret[self.running_var.name] = self.running_var
        return ret

    def set_states(self, states):
        self.set_params(states)
        self.running_mean.copy_from(states[self.running_mean.name])
        self.running_var.copy_from(states[self.running_var.name])


class Pooling2d(Layer):
    """
    Generate a Pooling 2d operator
    """

    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 is_max=True,
                 pad_mode="NOTSET"):
        """
        Args:
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            is_max (bool): is max pooling or avg pooling
            pad_mode (string): can be NOTSET, SAME_UPPER, or SAME_LOWER, where
                default value is NOTSET, which means explicit padding is used.
                SAME_UPPER or SAME_LOWER mean pad the input so that the output
                spatial size match the input. In case of odd number add the extra
                padding at the end for SAME_UPPER and at the beginning for SAME_LOWER.
        """
        super(Pooling2d, self).__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise TypeError("Wrong kernel_size type.")

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
            assert stride[0] > 0 or (kernel_size[0] == 1 and padding[0] == 0), (
                "stride[0]=0, but kernel_size[0]=%d, padding[0]=%d" %
                (kernel_size[0], padding[0]))
        else:
            raise TypeError("Wrong stride type.")

        self.odd_padding = (0, 0, 0, 0)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple) or isinstance(padding, list):
            if len(padding) == 2:
                self.padding = padding
            elif len(padding) == 4:
                _h_mask = padding[0] - padding[1]
                _w_mask = padding[2] - padding[3]
                # the odd paddding is the value that cannot be handled by the tuple padding (w, h) mode
                # so we need to firstly handle the input, then use the nomal padding method.
                self.odd_padding = (max(_h_mask, 0), max(-_h_mask, 0),
                                    max(_w_mask, 0), max(-_w_mask, 0))
                self.padding = (
                    padding[0] - self.odd_padding[0],
                    padding[2] - self.odd_padding[2],
                )
            else:
                raise TypeError("Wrong padding value.")

        self.is_max = is_max
        self.pad_mode = pad_mode

    def initialize(self, x):
        # if same pad mode, re-compute the padding
        if self.pad_mode in ("SAME_UPPER", "SAME_LOWER"):
            self.padding, self.odd_padding = utils.get_padding_shape(
                self.pad_mode, x.shape[2:], self.kernel_size, self.stride)

        # if same pad mode, re-compute the padding
        if self.pad_mode in ("SAME_UPPER", "SAME_LOWER"):
            self.padding, self.odd_padding = utils.get_padding_shape(
                self.pad_mode, x.shape[2:], self.kernel_size, self.stride)
            self.padding = [self.padding[0], self.padding[2]]

        _x = x
        if self.odd_padding != (0, 0, 0, 0):
            x_shape = list(x.data.shape())
            x_shape[2] += (self.odd_padding[0] + self.odd_padding[1])
            x_shape[3] += (self.odd_padding[2] + self.odd_padding[3])
            _x = Tensor(shape=x_shape, device=x.device)
            _x.set_value(0.0)

        if _x.device.id() == -1:
            self.handle = singa.PoolingHandle(
                _x.data,
                self.kernel_size,
                self.stride,
                self.padding,
                self.is_max,
            )
        else:
            self.handle = singa.CudnnPoolingHandle(
                _x.data,
                self.kernel_size,
                self.stride,
                self.padding,
                self.is_max,
            )

    def forward(self, x):
        y = autograd.pooling_2d(self.handle, x, self.odd_padding)
        return y


class MaxPool2d(Pooling2d):
    """
    Generate a Max Pooling 2d operator
    """

    def __init__(self, kernel_size, stride=None, padding=0, pad_mode="NOTSET"):
        """
        Args:
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            pad_mode (string): can be NOTSET, SAME_UPPER, or SAME_LOWER, where
                default value is NOTSET, which means explicit padding is used.
                SAME_UPPER or SAME_LOWER mean pad the input so that the output
                spatial size match the input. In case of odd number add the extra
                padding at the end for SAME_UPPER and at the beginning for SAME_LOWER.
        """
        super(MaxPool2d, self).__init__(kernel_size, stride, padding, True,
                                        pad_mode)


class AvgPool2d(Pooling2d):

    def __init__(self, kernel_size, stride=None, padding=0, pad_mode="NOTSET"):
        """
        Args:
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            pad_mode (string): can be NOTSET, SAME_UPPER, or SAME_LOWER, where
                default value is NOTSET, which means explicit padding is used.
                SAME_UPPER or SAME_LOWER mean pad the input so that the output
                spatial size match the input. In case of odd number add the extra
                padding at the end for SAME_UPPER and at the beginning for SAME_LOWER.
        """
        super(AvgPool2d, self).__init__(kernel_size, stride, padding, False,
                                        pad_mode)


class MaxPool1d(Pooling2d):
    """
    Generate a Max Pooling 1d operator
    """

    def __init__(self, kernel_size, stride=None, padding=0, pad_mode="NOTSET"):
        """
        Args:
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            pad_mode (string): can be NOTSET, SAME_UPPER, or SAME_LOWER, where
                default value is NOTSET, which means explicit padding is used.
                SAME_UPPER or SAME_LOWER mean pad the input so that the output
                spatial size match the input. In case of odd number add the extra
                padding at the end for SAME_UPPER and at the beginning for SAME_LOWER.
        """
        if stride is None:
            stride = kernel_size
        super(MaxPool1d, self).__init__((1, kernel_size), (1, stride),
                                        (0, padding), True, pad_mode)


class AvgPool1d(Pooling2d):
    """
    Generate a Avg Pooling 1d operator
    """

    def __init__(self, kernel_size, stride=None, padding=0, pad_mode="NOTSET"):
        """
        Args:
            kernel_size (int or tuple): kernel size for two direction of each
                axis. For example, (2, 3), the first 2 means will add 2 at the
                beginning and also 2 at the end for its axis.and if a int is
                accepted, the kernel size will be initiated as (int, int)
            stride (int or tuple): stride, the logic is the same as kernel size.
            padding (int): tuple, list or None, padding, the logic is the same
                as kernel size. However, if you set pad_mode as "SAME_UPPER" or
                "SAME_LOWER" mode, you can set padding as None, and the padding
                will be computed automatically.
            pad_mode (string): can be NOTSET, SAME_UPPER, or SAME_LOWER, where
                default value is NOTSET, which means explicit padding is used.
                SAME_UPPER or SAME_LOWER mean pad the input so that the output
                spatial size match the input. In case of odd number add the extra
                padding at the end for SAME_UPPER and at the beginning for SAME_LOWER.
        """
        if stride is None:
            stride = kernel_size
        super(AvgPool1d, self).__init__((1, kernel_size), (1, stride),
                                        (0, padding), False, pad_mode)


class RNN_Base(Layer):

    def step_forward(self,
                     x=None,
                     h=None,
                     c=None,
                     Wx=None,
                     Wh=None,
                     Bx=None,
                     Bh=None,
                     b=None):
        raise NotImplementedError


class RNN(RNN_Base):
    """
    Generate a RNN operator
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        nonlinearity="tanh",
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
    ):
        """
        Args:
            input_size (int):  The number of expected features in the input x
            hidden_size (int): The number of features in the hidden state h
            num_layers (int):  Number of recurrent layers. Default: 1
            nonlinearity (string): The non-linearity to use. Default: 'tanh'
            bias (bool):  If False, then the layer does not use bias weights.
                Default: True
            batch_first (bool):  If True, then the input and output tensors
                are provided as (batch, seq, feature). Default: False
            dropout (float): If non-zero, introduces a Dropout layer on the
                outputs of each RNN layer except the last layer, with dropout
                probability equal to dropout. Default: 0
            bidirectional (bool): If True, becomes a bidirectional RNN.
                Default: False
        """
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

    def initialize(self, xs, h0):
        Wx_name = self.name + Layer.sep + 'Wx'
        Wx_shape = (self.input_size, self.hidden_size)
        self.Wx = Tensor(shape=Wx_shape,
                         requires_grad=True,
                         stores_grad=True,
                         name=Wx_name)
        self.Wx.gaussian(0.0, 1.0)

        Wh_name = self.name + Layer.sep + 'Wh'
        Wh_shape = (self.hidden_size, self.hidden_size)
        self.Wh = Tensor(shape=Wh_shape,
                         requires_grad=True,
                         stores_grad=True,
                         name=Wh_name)
        self.Wh.gaussian(0.0, 1.0)

        b_name = self.name + Layer.sep + 'b'
        b_shape = (self.hidden_size,)
        self.b = Tensor(shape=b_shape,
                        requires_grad=True,
                        stores_grad=True,
                        name=b_name)
        self.b.set_value(0.0)

    def forward(self, xs, h0):
        # xs: a tuple or list of input tensors
        if not isinstance(xs, tuple):
            xs = tuple(xs)
        inputs = xs + (h0,)
        self.device_check(*inputs)
        # self.device_check(inputs[0], *self.params)
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
        y2 = autograd.matmul(h, Wh)
        y1 = autograd.matmul(x, Wx)
        y = autograd.add(y2, y1)
        y = autograd.add_bias(y, b, axis=0)
        if self.nonlinearity == "tanh":
            y = autograd.tanh(y)
        elif self.nonlinearity == "relu":
            y = autograd.relu(y)
        else:
            raise ValueError
        return y

    def get_params(self):
        return {
            self.Wx.name: self.Wx,
            self.Wh.name: self.Wh,
            self.b.name: self.b
        }

    def set_params(self, parameters):
        self.Wx.copy_from(parameters[self.Wx.name])
        self.Wh.copy_from(parameters[self.Wh.name])
        self.b.copy_from(parameters[self.b.name])


class LSTM(RNN_Base):
    """
    Generate a LSTM operator
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        nonlinearity="tanh",
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
    ):
        """
        Args:
            input_size (int):  The number of expected features in the input x
            hidden_size (int): The number of features in the hidden state h
            num_layers (int):  Number of recurrent layers. Default: 1
            nonlinearity (string): The non-linearity to use. Default: 'tanh'
            bias (bool):  If False, then the layer does not use bias weights.
                Default: True
            batch_first (bool):  If True, then the input and output tensors
                are provided as (batch, seq, feature). Default: False
            dropout (float): If non-zero, introduces a Dropout layer on the
                outputs of each RNN layer except the last layer, with dropout
                probability equal to dropout. Default: 0
            bidirectional (bool): If True, becomes a bidirectional RNN.
                Default: False
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

    def initialize(self, xs, h0_c0):
        # 1. Wx_i input,  Bx_i
        # 2. Wx_f forget, Bx_f
        # 3. Wx_o output, Bx_o
        # 4. Wx_g candidate, Bx_g
        prefix = self.name + Layer.sep
        Wx_shape = (self.input_size, self.hidden_size)
        self.Wx_i = Tensor(shape=Wx_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=prefix + 'Wx_i')
        self.Wx_f = Tensor(shape=Wx_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=prefix + 'Wx_f')
        self.Wx_o = Tensor(shape=Wx_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=prefix + 'Wx_o')
        self.Wx_g = Tensor(shape=Wx_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=prefix + 'Wx_g')

        Wh_shape = (self.hidden_size, self.hidden_size)
        self.Wh_i = Tensor(shape=Wh_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=prefix + 'Wh_i')
        self.Wh_f = Tensor(shape=Wh_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=prefix + 'Wh_f')
        self.Wh_o = Tensor(shape=Wh_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=prefix + 'Wh_o')
        self.Wh_g = Tensor(shape=Wh_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=prefix + 'Wh_g')
        [
            w.gaussian(0.0, 0.01) for w in [
                self.Wx_i, self.Wx_f, self.Wx_o, self.Wx_g, self.Wh_i,
                self.Wh_f, self.Wh_o, self.Wh_g
            ]
        ]

        Bx_shape = (self.hidden_size,)
        self.Bx_i = Tensor(shape=Bx_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=prefix + 'Bx_i')
        self.Bx_f = Tensor(shape=Bx_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=prefix + 'Bx_f')
        self.Bx_o = Tensor(shape=Bx_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=prefix + 'Bx_o')
        self.Bx_g = Tensor(shape=Bx_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=prefix + 'Bx_g')
        self.Bh_i = Tensor(shape=Bx_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=prefix + 'Bh_i')
        self.Bh_f = Tensor(shape=Bx_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=prefix + 'Bh_f')
        self.Bh_o = Tensor(shape=Bx_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=prefix + 'Bh_o')
        self.Bh_g = Tensor(shape=Bx_shape,
                           requires_grad=True,
                           stores_grad=True,
                           name=prefix + 'Bh_g')
        [
            b.set_value(0.0) for b in [
                self.Bx_i, self.Bx_f, self.Bx_o, self.Bx_g, self.Bh_i,
                self.Bh_f, self.Bh_o, self.Bh_g
            ]
        ]

    def forward(self, xs, h0_c0):
        # xs: a tuple or list of input tensors
        # h0_c0: a tuple of (h0, c0)
        h0, c0 = h0_c0
        if not isinstance(xs, list):
            xs = list(xs)
        inputs = xs + list((h0, c0))
        self.device_check(*inputs)
        self.device_check(inputs[0], *[s for k, s in self.get_states().items()])
        batchsize = xs[0].shape[0]
        out = []
        h, c = self.step_forward(xs[0], h0, c0)
        out.append(h)
        for x in xs[1:]:
            assert x.shape[0] == batchsize
            h, c = self.step_forward(x, h, c)
            out.append(h)
        return out, h, c

    def step_forward(self, x, h, c):
        # input
        y1 = autograd.matmul(x, self.Wx_i)
        y1 = autograd.add_bias(y1, self.Bx_i, axis=0)
        y2 = autograd.matmul(h, self.Wh_i)
        y2 = autograd.add_bias(y2, self.Bh_i, axis=0)
        i = autograd.add(y1, y2)
        i = autograd.sigmoid(i)

        # forget
        y1 = autograd.matmul(x, self.Wx_f)
        y1 = autograd.add_bias(y1, self.Bx_f, axis=0)
        y2 = autograd.matmul(h, self.Wh_f)
        y2 = autograd.add_bias(y2, self.Bh_f, axis=0)
        f = autograd.add(y1, y2)
        f = autograd.sigmoid(f)

        # output
        y1 = autograd.matmul(x, self.Wx_o)
        y1 = autograd.add_bias(y1, self.Bx_o, axis=0)
        y2 = autograd.matmul(h, self.Wh_o)
        y2 = autograd.add_bias(y2, self.Bh_o, axis=0)
        o = autograd.add(y1, y2)
        o = autograd.sigmoid(o)

        y1 = autograd.matmul(x, self.Wx_g)
        y1 = autograd.add_bias(y1, self.Bx_g, axis=0)
        y2 = autograd.matmul(h, self.Wh_g)
        y2 = autograd.add_bias(y2, self.Bh_g, axis=0)
        g = autograd.add(y1, y2)
        g = autograd.tanh(g)

        cout1 = autograd.mul(f, c)
        cout2 = autograd.mul(i, g)
        cout = autograd.add(cout1, cout2)

        hout = autograd.tanh(cout)
        hout = autograd.mul(o, hout)
        return hout, cout

    def get_params(self):
        ret = {}
        for w in [
                self.Wx_i, self.Wx_f, self.Wx_o, self.Wx_g, self.Wh_i,
                self.Wh_f, self.Wh_o, self.Wh_g
        ]:
            ret[w.name] = w

        for b in [
                self.Bx_i, self.Bx_f, self.Bx_o, self.Bx_g, self.Bh_i,
                self.Bh_f, self.Bh_o, self.Bh_g
        ]:
            ret[b.name] = b
        return ret

    def set_params(self, parameters):
        for w in [
                self.Wx_i, self.Wx_f, self.Wx_o, self.Wx_g, self.Wh_i,
                self.Wh_f, self.Wh_o, self.Wh_g
        ]:
            w.copy_from(parameters[w.name])

        for b in [
                self.Bx_i, self.Bx_f, self.Bx_o, self.Bx_g, self.Bh_i,
                self.Bh_f, self.Bh_o, self.Bh_g
        ]:
            b.copy_from(parameters[b.name])


''' layers without params or states
'''


class ReLU(Layer):
    """
    Generate a ReLU operator
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return autograd.relu(x)


class Add(Layer):
    """
    Generate a Add operator
    """

    def __init__(self):
        super(Add, self).__init__()

    def forward(self, a, b):
        return autograd.add(a, b)


class Flatten(Layer):
    """
    Generate a Flatten operator
    """

    def __init__(self, axis=1):
        super(Flatten, self).__init__()
        self.axis = axis

    def forward(self, x):
        return autograd.flatten(x, self.axis)


class SoftMaxCrossEntropy(Layer):
    """
    Generate a SoftMaxCrossEntropy operator
    """

    def __init__(self):
        super(SoftMaxCrossEntropy, self).__init__()

    def forward(self, x, t):
        return autograd.softmax_cross_entropy(x, t)


class Dropout(Layer):
    """
    Generate a Dropout operator
    """

    def __init__(self, ratio=0.5):
        super(Dropout, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        return autograd.dropout(x, self.ratio)


class Cat(Layer):
    """
    Generate a Cat Operator
    """

    def __init__(self, axis=0):
        super(Cat, self).__init__()
        self.axis = axis

    def forward(self, xs):
        return autograd.cat(xs, self.axis)


class Reshape(Layer):
    """
    Generate a Reshape Operator
    """

    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x, shape):
        return autograd.reshape(x, shape)


class CudnnRNN(Layer):
    """ `CudnnRNN` class implements with c++ backend and run the operation
          directly on cuDNN
        While `RNN` class implements with high level singa API
    """

    def __init__(self,
                 hidden_size,
                 activation="tanh",
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0,
                 bidirectional=False,
                 rnn_mode="lstm",
                 return_sequences=False):
        """
            Args:
                hidden_size: hidden feature dim
                rnn_mode: accepted value: "vanilla", "tanh", "relu",  "lstm", "gru"
        """
        assert singa.USE_CUDA, "Not able to run without CUDA"
        super(CudnnRNN, self).__init__()

        self.rnn_mode = rnn_mode
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = 1 if bidirectional else 0
        self.return_sequences = return_sequences
        self.batch_first = batch_first

        # GPU parameter
        # cudnn_rnn_mode: 0 - RNN RELU, 1 - RNN TANH, 2 - LSTM, 3 - GRU
        if self.rnn_mode == "lstm":
            self.cudnn_rnn_mode = 2
        elif self.rnn_mode == "vanilla" or self.rnn_mode == "tanh":
            self.cudnn_rnn_mode = 1
        elif self.rnn_mode == "relu":
            self.cudnn_rnn_mode = 0
        elif self.rnn_mode == "gru":
            self.cudnn_rnn_mode = 3

    def initialize(self, x, hx=None, cx=None):
        if self.batch_first:
            x = x.transpose((1, 0, 2))
        self.input_size = x.shape[1]

        # GPU handle
        self.handle = singa.CudnnRNNHandle(x.data,
                                           self.hidden_size,
                                           mode=self.cudnn_rnn_mode,
                                           num_layers=self.num_layers,
                                           dropout=self.dropout,
                                           bidirectional=self.bidirectional)

        w_name = self.name + Layer.sep + 'W'
        self.W = Tensor(shape=(self.handle.weights_size,),
                        requires_grad=True,
                        stores_grad=True,
                        name=w_name,
                        device=x.device)
        self.W.gaussian(0, 1)

    def forward(self, x, hx=None, cx=None):

        self.device_check(x, self.W)
        if self.batch_first:
            x = x.transpose((1, 0, 2))

        batch_size = x.shape[1]
        directions = 2 if self.bidirectional else 1
        if hx == None:
            hx = Tensor(shape=(self.num_layers * directions, batch_size,
                                           self.hidden_size),
                                    requires_grad=False,
                                    stores_grad=False,
                                    device=x.device).set_value(0.0)
        if cx == None:
            cx = Tensor(shape=(self.num_layers * directions, batch_size,
                                           self.hidden_size),
                                    requires_grad=False,
                                    stores_grad=False,
                                    device=x.device).set_value(0.0)

        # outputs returned is list
        #   inputs has shape of {sequence length, batch size, feature size}
        y = autograd._RNN(self.handle,
                          return_sequences=self.return_sequences)(x, hx, cx,
                                                                  self.W)[0]
        if self.return_sequences and self.batch_first:
            #   outputs has shape of {sequence length, batch size, hidden size}
            y = y.transpose((1, 0, 2))  # to {bs, seq, hid}
        return y

    def get_params(self):
        return {self.W.name: self.W}

    def set_params(self, parameters):
        self.set_attribute(self.W, parameters[self.W.name])


''' import autograd at the end to resolve circular import
'''
from singa import autograd
