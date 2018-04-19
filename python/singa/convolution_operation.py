from . import tensor
from . import layer


class Convolution2D(tensor.Operation):
    def __init__(self, name, nb_kernels, kernel=3, stride=1, border_mode='same',
                 cudnn_prefer='fastest', workspace_byte_limit=1024,
                 data_format='NCHW', use_bias=True, W_specs=None, b_specs=None,
                 pad=None):
        self.PyLayer = layer.Conv2D(name, nb_kernels, kernel=kernel, stride=stride, border_mode=border_mode,
                 cudnn_prefer=cudnn_prefer, workspace_byte_limit=workspace_byte_limit,
                 data_format=data_format, use_bias=use_bias, W_specs=W_specs, b_specs=b_specs,
                 pad=pad, input_sample_shape=None)

    def __call__(self, *x):
        self.PyLayer.setup(x.shape)

        param_data = self.PyLayer.layer.param_values()
        if len(param_data) == 2:
            w = tensor.Tensor(data=param_data[0], requires_grad=True, stores_grad=True)
            b = tensor.Tensor(data=param_data[1], requires_grad=True, stores_grad=True)
            xs = [x, w, b]
        elif len(param_data) == 1:
            w = tensor.Tensor(data=param_data[0], requires_grad=True, stores_grad=True)
            xs = [x, w]
        return self._do_forward(*xs)

    def forward(self, x):
        return self.PyLayer.layer.Forward(True, x)

    def backward(self, dy):
        ret = self.PyLayer.layer.Backward(True, dy)
        return (ret[0],)+ret[1]