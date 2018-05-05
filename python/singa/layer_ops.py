from singa import tensor
from singa import layer
from singa.proto import model_pb2


class Conv2D(tensor.Operation):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,**kwargs):

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

        '''
        How to match Keras:

        in Keras conv2d, self.kernel record how to generate kernel (shape,initializer,name,regularizer,constraint),
        it can be interpret to
        shape -> kernel+input_sample_shape[0](nb_channels)+nb_kernels,
        initializer, name, regularizer, constraint -> W_specs.
        '''
        self.PyLayer = layer.Conv2D(name, nb_kernels=out_channels, kernel=kernel_size, stride=stride, border_mode=border_mode,
                 cudnn_prefer=cudnn_prefer, workspace_byte_limit=workspace_byte_limit,
                 data_format=data_format, use_bias=bias, W_specs=W_specs, b_specs=b_specs,
                 pad=padding, input_sample_shape=input_sample_shape)


    def __call__(self, x):
        if not self.PyLayer.has_setup:
            self.PyLayer.setup(x.shape[1:])
        param_data = self.PyLayer.layer.param_values()
        if not hasattr(self, 'w'):
            self.w = tensor.Tensor(data=param_data[0], requires_grad=True, stores_grad=True)
            self.w.gaussian(0.0, 0.1)  # TODO realize other initialization method according to W_specs
        
        xs = [x, self.w]

        if len(param_data) == 2:
            self.b = tensor.Tensor(data=param_data[1], requires_grad=True, stores_grad=True)
            self.b.set_value(0.0)  # TODO realize other initialization method according to b_specs
            xs.append(self.b)

        xs = tuple(xs)
        return self._do_forward(*xs)

    def forward(self, flag=True,*xs):
        if flag is True:
            return self.PyLayer.layer.Forward(4, xs[0])
        else:
            return self.PyLayer.layer.Forward(8, xs[0])

    def backward(self, dy):
        ret = self.PyLayer.layer.Backward(0, dy)
        return (ret[0],)+ret[1]


class MaxPooling2D(tensor.Operation):
    def __init__(self, name, kernel=3, stride=2, border_mode='same', pad=None,
                 data_format='NCHW', input_sample_shape=None):

        self.PyLayer = layer.Pooling2D(name, model_pb2.PoolingConf.MAX,
                                           kernel, stride, border_mode,
                                           pad, data_format, input_sample_shape)

    def __call__(self, x):
        if not self.PyLayer.has_setup:
            self.PyLayer.setup(x.shape[1:])
        return self._do_forward(x)

    def forward(self, x):
        return self.PyLayer.layer.Forward(4, x)

    def backward(self, dy):
        return self.PyLayer.layer.Backward(0, dy)[0]


class Activation(tensor.Operation):
    def __init__(self,name, mode='relu',input_sample_shape=None):
        self.PyLayer = layer.Activation(name, mode, input_sample_shape)

    def __call__(self, x):
        if not self.PyLayer.has_setup:
            self.PyLayer.setup(x.shape[1:])
        return self._do_forward(x)

    def forward(self, x):
        return self.PyLayer.layer.Forward(4, x)

    def backward(self, dy):
        return self.PyLayer.layer.Backward(0, dy)[0]


class Flatten(tensor.Operation):
    def __init__(self, name, axis=1, input_sample_shape=None):
        self.PyLayer = layer.Flatten(name, axis, input_sample_shape)

    def __call__(self, x):
        if not self.PyLayer.has_setup:
            self.PyLayer.setup(x.shape[1:])
        return self._do_forward(x)

    def forward(self, x):
        return self.PyLayer.layer.Forward(4, x)

    def backward(self, dy):
        return self.PyLayer.layer.Backward(0, dy)[0]


class Dense(tensor.Operation):
    def __init__(self, name, num_output, use_bias=True,
                     W_specs=None, b_specs=None,
                     W_transpose=False, input_sample_shape=None):

        self.PyLayer = layer.Dense(name, num_output=num_output, use_bias=use_bias,
                     W_specs=W_specs, b_specs=b_specs,
                     W_transpose=W_transpose, input_sample_shape=input_sample_shape)

    def __call__(self, x):
        if not self.PyLayer.has_setup:
            self.PyLayer.setup(x.shape[1:])

        param_data = self.PyLayer.layer.param_values()

        if not hasattr(self, 'w'):
            self.w = tensor.Tensor(data=param_data[0], requires_grad=True, stores_grad=True)
            self.w.gaussian(0.0, 0.1)  # TODO realize other initialization method according to W_specs

        xs = [x, self.w]

        if len(param_data) == 2:
            self.b = tensor.Tensor(data=param_data[1], requires_grad=True, stores_grad=True)
            self.b.set_value(0.0)  # TODO realize other initialization method according to b_specs
            xs.append(self.b)

        xs = tuple(xs)
        return self._do_forward(*xs)

    def forward(self, *xs):
        return self.PyLayer.layer.Forward(4, xs[0])

    def backward(self, dy):
        ret = self.PyLayer.layer.Backward(0, dy)
        return (ret[0],)+ret[1]








