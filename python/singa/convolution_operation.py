from singa import tensor
from singa import layer
from singa.proto import model_pb2


def ctensor2numpy(x):
    '''
    // For test use.


    To be used in SoftMax Operation.
    Convert a singa_tensor to numpy_tensor.
    '''
    np_array = x.GetFloatValue(int(x.Size()))
    return np_array.reshape(x.shape())

class Convolution2D(tensor.Operation):
    def __init__(self, name, nb_kernels, kernel=3, stride=1, border_mode='same',
                 cudnn_prefer='fastest', workspace_byte_limit=1024,
                 data_format='NCHW', use_bias=True, W_specs=None, b_specs=None,
                 pad=None,input_sample_shape=None):
        '''
        How to match Keras:

        in Keras conv2d, self.kernel record how to generate kernel (shape,initializer,name,regularizer,constraint),
        it can be interpret to
        shape -> kernel+input_sample_shape[0](nb_channels)+nb_kernels,
        initializer, name, regularizer, constraint -> W_specs.
        '''
        self.PyLayer = layer.Conv2D(name, nb_kernels, kernel=kernel, stride=stride, border_mode=border_mode,
                 cudnn_prefer=cudnn_prefer, workspace_byte_limit=workspace_byte_limit,
                 data_format=data_format, use_bias=use_bias, W_specs=W_specs, b_specs=b_specs,
                 pad=pad, input_sample_shape=input_sample_shape)

    def __call__(self, x):
        if not self.PyLayer.has_setup:
            self.PyLayer.setup(x.shape[1:])
        param_data = self.PyLayer.layer.param_values()
        if len(param_data) == 2:
            w = tensor.Tensor(data=param_data[0], requires_grad=True, stores_grad=True)
            b = tensor.Tensor(data=param_data[1], requires_grad=True, stores_grad=True)
            w.gaussian(0.0, 0.1)
            b.set_value(0.0)
            xs = tuple([x, w, b])
        else:
            w = tensor.Tensor(data=param_data[0], requires_grad=True, stores_grad=True)
            w.gaussian(0.0, 0.1)
            xs = tuple([x, w])
        return self._do_forward(*xs)

    def forward(self, *xs):
        return self.PyLayer.layer.Forward(4, xs[0])  #how ktrain works?  flag & ktrain.

    def backward(self, dy):
        ret = self.PyLayer.layer.Backward(True, dy)
        return (ret[0],)+ret[1]


x = tensor.Tensor(shape=(1, 1, 3, 3), requires_grad=True, stores_grad=True)
x.gaussian(1,0.0)
layer_1= Convolution2D('conv1',4)
y= layer_1(x)[0]
z= layer_1._do_backward(y.data)
a=ctensor2numpy(y.data)




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
        return self.PyLayer.layer.Backward(True, dy)[0]   # how backward() return?

x = tensor.Tensor(shape=(1, 1, 3, 3), requires_grad=True, stores_grad=True)
x.gaussian(1,0.0)
layer_1= MaxPooling2D('pooling1')
y= layer_1(x)[0]
z= layer_1._do_backward(y.data)
a=ctensor2numpy(y.data)


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
        return self.PyLayer.layer.Backward(True, dy)[0]


x = tensor.Tensor(shape=(1, 1, 3, 3), requires_grad=True, stores_grad=True)
x.gaussian(-1,0.0)
layer_1= Activation('relu1')
y= layer_1(x)[0]
z= layer_1._do_backward(y.data)
a=ctensor2numpy(y.data)


class Flatten(tensor.Operation):
    def __init__(self, name, axis=1, input_sample_shape=None):
        self.PyLayer = layer.Flatten(name, axis, input_sample_shape)

    def __call__(self, x):
        if not self.PyLayer.has_setup:
            self.PyLayer.setup(x.shape)
        return self._do_forward(x)

    def forward(self, x):
        return self.PyLayer.layer.Forward(4, x)

    def backward(self, dy):
        return self.PyLayer.layer.Backward(True, dy)[0]

x = tensor.Tensor(shape=(1, 1, 3, 3), requires_grad=True, stores_grad=True)
x.gaussian(-1,0.0)
layer_1= Flatten('flatten')
y= layer_1(x)[0]
z= layer_1._do_backward(y.data)
a=ctensor2numpy(y.data)





