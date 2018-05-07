from singa import tensor
from singa import layer
from singa.proto import model_pb2


class Conv2d(tensor.Operation):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):

        name='Conv2d'
        border_mode = 'same'
        cudnn_prefer = 'fastest'
        workspace_byte_limit = 1024
        data_format = 'NCHW'
        W_specs = None
        b_specs = None
        input_sample_shape=None

        allowed_kwargs = {'name':name,
                          'border_mode':border_mode,
                          'cudnn_prefer':cudnn_prefer,
                          'workspace_byte_limit':workspace_byte_limit,
                          'data_format':data_format,
                          'W_specs':W_specs,
                          'b_specs':b_specs,
                          'input_sample_shape':input_sample_shape
                          }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)
            else:
                allowed_kwargs[kwarg] = kwargs[kwarg]

        self.W_specs=W_specs
        self.b_specs=b_specs

        if padding == 0:
            pad = None
        else:
            pad = padding

        if dilation != 1 or groups != 1:
            raise ValueError('Not implemented yet')

        self.PyLayer = layer.Conv2D(name, nb_kernels=out_channels, kernel=kernel_size, stride=stride, border_mode=border_mode,
                 cudnn_prefer=cudnn_prefer, workspace_byte_limit=workspace_byte_limit,
                 data_format=data_format, use_bias=bias, W_specs=self.W_specs, b_specs=self.b_specs,
                 pad=pad, input_sample_shape=input_sample_shape)

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
            self.w = tensor.Tensor(data=param_data[0], requires_grad=True, stores_grad=True)
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
                self.b = tensor.Tensor(data=param_data[1], requires_grad=True, stores_grad=True)
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
                else:
                    self.b.set_value(0.0)

            xs.append(self.b)

        xs = tuple(xs)
        return self._do_forward(*xs)

    def forward(self, *xs):
        return self.PyLayer.layer.Forward(self.flag, xs[0])

    def backward(self, dy):
        ret = self.PyLayer.layer.Backward(0, dy)
        return (ret[0],)+ret[1]


class MaxPool2d(tensor.Operation):
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


class ReLU(tensor.Operation):
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

    def forward(self, flag=True, *xs):
        return self.PyLayer.layer.Forward(self.flag, xs[0])

    def backward(self, dy):
        return self.PyLayer.layer.Backward(0, dy)[0]


class Flatten(tensor.Operation):
    def __init__(self, name, axis=1, input_sample_shape=None):
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


class Linear(tensor.Operation):
    def __init__(self, in_features, out_features, bias=True, **kwargs):

        name = 'Linear'
        W_transpose=False
        W_specs = None
        b_specs = None
        input_sample_shape = in_features

        allowed_kwargs = {'name': name,
                          'W_transpose': W_transpose,
                          'W_specs': W_specs,
                          'b_specs': b_specs,
                          'input_sample_shape': input_sample_shape
                          }

        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)
            else:
                allowed_kwargs[kwarg] = kwargs[kwarg]

        self.W_specs = W_specs
        self.b_specs = b_specs

        self.PyLayer = layer.Dense(name, num_output=out_features, use_bias=bias,
                     W_specs=self.W_specs, b_specs=self.b_specs,
                     W_transpose=W_transpose, input_sample_shape=input_sample_shape)

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
            self.w = tensor.Tensor(data=param_data[0], requires_grad=True, stores_grad=True)
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
                self.b = tensor.Tensor(data=param_data[1], requires_grad=True, stores_grad=True)
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
                else:
                    self.b.set_value(0.0)

            xs.append(self.b)

        xs = tuple(xs)
        return self._do_forward(*xs)

    def forward(self, *xs):
        return self.PyLayer.layer.Forward(self.flag, xs[0])

    def backward(self, dy):
        ret = self.PyLayer.layer.Backward(0, dy)
        return (ret[0],)+ret[1]
