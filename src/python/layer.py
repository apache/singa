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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from sets import Set


class Layer(object):
    """Base Python layer class.

    Usages:
        1. construct layer with input_shapes, goto 3; construct layer without
            input_shapes, goto 2
        2. call setup to create the parameters and setup other meta fields
        3. call forward or access layer members
        4. call backward and get parameters for update
    """

    def __init__(self, name, **kwargs):
        self.name = name
        self.layer = None  # layer converted by swig
        self.conf = None  # protobuf object for layer hyper-parameters
        self.has_setup = False

    def setup(self, in_shapes):
        """Call the setup function of c++ layers.

        Args:
            in_shapes: if the layer accepts a single input Tensor, in_shapes is
                a single tuple specifying the inpute Tensor shape; if the layer
                accepts multiple input Tensor (e.g., concatenation layer),
                in_shapes is a tuple of tuples, each for one input Tensor shape
        """
        if self.has_setup:
            return
        self.has_setup = True
        self.layer.Setup(in_shapes, self.conf.SerializeToString())

    def get_output_shapes(self):
        assert self.has_setup, 'Must call setup() before get_output_shapes()'
        self.layer.get_output_shapes()

    def forward(self, flag, inputs):
        assert self.has_setup, 'Must call setup() before forward()'
        if isinstance(inputs, tuple):
            return self.cpp_layer.forward_multiple_inputs(flag, inputs)
        else:
            return self.cpp_layer.foward_single_input(flag, inputs)

    def backward(self, flag, grads):
        if isinstance(grads, tuple):
            return self.cpp_layer.backward_multiple_grads(flag, grads)
        else:
            return self.cpp_layer.backward_single_grad(flag, grads)

    def to_device(self, device):
        self.layer.ToDevice(device)

    def as_type(self, dtype):
        self.layer.AsType(dtype)

    def __copy__(self):
        pass

    def __deepcopy__(self):
        pass


class Conv2D(Layer):
    def __init__(self, nb_kernels, kernel=3, stride=1, border_mode='valid',
                 engine='cudnn', cudnn_prefer='fatest', data_format='NCHW',
                 use_bias=True, pad=None, W_specs=None, b_specs=None, name=None,
                 input_shape=None):
        """Construct a layer for 2D convolution.

        Args:
            nb_kernels (int): num of the channels (kernels) of the input Tensor
            kernel: an integer or a pair of integers for kernel height and width
            stride: an integer or a pair of integers for stride height and width
            border_mode (string): padding mode, case in-sensitive,
                'valid' -> padding is 0 for height and width
                'same' -> padding is half of the kernel (floor)
            engine (string): implementation engin, could be 'cudnn'
                (case insensitive)
            cudnn_prefer (string): the preferred algorithm for cudnn convolution
                which could be 'fatest', 'autotune', 'limited_workspace' and
                'no_workspace'
            data_format (string): either 'NCHW' or 'NHWC'
            use_bias (bool): True or False
            pad: an integer or a pair of integers for padding height and width
            W_specs (dict): used to specify the weight matrix specs, fields
                include,
                'name' for parameter name
                'lr_mult' for learning rate multiplier
                'decay_mult' for weight decay multiplier
                'init' for init method, which could be 'gaussian', 'uniform',
                'xavier' and ''
                'std', 'mean', 'high', 'low' for corresponding init methods
                'clamp' for gradient constraint, value is scalar
                'regularizer' for regularization, currently support 'l2'
            b_specs (dict): hyper-parameters for bias vector, similar as W_specs
            name (string): layer name.
            input_shape: 3d tuple for the shape of the input Tensor without the
                batchsize, e.g., (channel, height, width) or
                (height, width, channel)
        """
        super(Conv2D, self).__init__(name)
        conf = self.conf.ConvolutionConf()
        conf.num_output = nb_kernels
        conf = _set_kernel_stride_pad(kernel, stride, border_mode, pad)
        conf.bias_term = use_bias
        # TODO(wangwei) enable data format for cpp code
        # conf.data_format = data_format
        if W_specs is not None:
            self.conf.add_param(_construct_param_specs_from_dict(W_specs))
            self.param_specs.append(W_specs)
        if b_specs is not None:
            self.conf.add_param(_construct_param_specs_from_dict(b_specs))
            self.param_specs.append(b_specs)

        _check_engine(engine, ['cudnn'])
        self.layer = _create_layer(engine, 'Convolution')
        if input_shape is not None:
            self.setup(input_shape)


class Conv1D(Conv2D):

    def __init__(self, in_channles, out_channels, kernel=3, stride=1,
                 border_mode='valid', engine='cudnn', cudnn_prefer='fatest',
                 use_bias=True, pad=None, W_specs=None, b_specs=None,
                 name=None, input_shape=None):
        """Construct a layer for 1D convolution.

        Most of the args are the same as those for Conv2D except the kernel,
        stride, pad, which is a scalar instead of a tuple.
        input_shape is a tuple with a single value for the input feature length
        """
        pad2 = None
        if pad is not None:
            pad2 = (pad, 0)
        if input_shape is not None:
            input_shape = (1, input_shape[0], 1)
        super(Conv1D, self).__init__(in_channles, out_channels, (kernel, 1),
                                     (stride, 0), border_mode, engine,
                                     cudnn_prefer, use_bias, pad2, W_specs,
                                     b_specs, name, input_shape)


class Pooling2D(Layer):
    def __init__(self, mode, kernel=3, stride=2, border_mode='valid', pad=None,
                 data_format='NCHW', engine='cudnn', name=None,
                 input_shape=None):
        super(MaxPooling2D, Layer).__init__(name)
        conf = self.conf.pooling_conf()
        conf = _set_kernel_stride_pad(conf, kernel, stride, border_mode, pad)
        conf.pool = mode
        _check_engine(engine, ['cudnn'])
        self.layer = _create_layer(engine, 'Pooling')


class MaxPooling2D(Pooling2D):
    def __init__(self, kernel=3, stride=2, border_mode='valid', pad=None,
                 data_format='NCHW', engine='cudnn', name=None,
                 input_shape=None):
        super(MaxPooling2D, Pooling2D).__init__(PoolingMethod_MAX, kernel,
                                                stride, border_mode, pad,
                                                data_format, engine, name,
                                                input_shape)


class AvgPooling2D(Pooling2D):
    def __init__(self, kernel=3, stride=2, border_mode='valid', pad=None,
                 data_format='NCHW', engine='cudnn', name=None,
                 input_shape=None):
        super(MaxPooling2D, Pooling2D).__init__(PoolingMethod_AVG, kernel,
                                                stride, border_mode, pad,
                                                data_format, engine, name,
                                                input_shape)


class MaxPooling1D(MaxPooling2D):
    def __init__(self, kernel=3, stride=2, border_mode='valid', pad=None,
                 data_format='NCHW', engine='cudnn', name=None,
                 input_feature_length=None):
        pad2 = None
        if pad is not None:
            pad2 = (pad, 0)
        if input_feature_length is not None:
            input_shape = (1, input_feature_length, 1)
        else:
            input_shape = None
        super(MaxPooling1D, MaxPooling2D).__init__((kernel, 1), (stride, 0),
                                                   border_mode, pad2,
                                                   data_format, engine, name,
                                                   input_shape)


class AvgPooling1D(AvgPooling2D):
    def __init__(self, kernel=3, stride=2, border_mode='valid', pad=None,
                 data_format='NCHW', engine='cudnn', name=None,
                 input_feature_length=None):
        """input_feature_length is a scalar value"""
        pad2 = None
        if pad is not None:
            pad2 = (pad, 0)
        if input_feature_length is not None:
            input_shape = (1, input_feature_length, 1)
        else:
            input_shape = None

        super(AvgPooling1D, AvgPooling2D).__init__((kernel, 1), (stride, 0),
                                                   border_mode, pad2,
                                                   data_format, engine, name,
                                                   input_shape)


class BatchNormalization(Layer):
    # TODO(wangwei) add mode and epsilon arguments
    def __init__(self, momentum=0.9, engine='cudnn', beta_specs=None,
                 gamma_specs=None, name=None, input_shape=None):
        """Batch-normalization.

        Args:
            momentum (float): for running average mean and variance.
            beta_specs (dict): dictionary includes the fields for the beta
                param:
                'name' for parameter name
                'lr_mult' for learning rate multiplier
                'decay_mult' for weight decay multiplier
                'init' for init method, which could be 'gaussian', 'uniform',
                'xavier' and ''
                'std', 'mean', 'high', 'low' for corresponding init methods
                'clamp' for gradient constraint, value is scalar
                'regularizer' for regularization, currently support 'l2'
            gamma_specs (dict): similar to beta_specs, but for the gamma param.
            name (string): layer name
            input_shape (tuple): with at least one integer
        """
        super(BatchNormalization, Layer).__init__(name)
        conf = self.conf.batchnorm_conf()
        conf.factor = momentum
        self.conf.add_param(_construct_param_specs_from_dict(beta_specs))
        self.conf.add_param(_construct_param_specs_from_dict(gamma_specs))
        self.param_specs.append(beta_specs)
        self.param_specs.append(gamma_specs)
        _check_engine(engine, ['cudnn'])
        self.layer = _create_layer(engine, 'BatchNorm')
        if input_shape is not None:
            self.setup(input_shape)


class LRN(Layer):
    def __init__(self, size=5, alpha=1, beta=0.75, mode='across_channel', k=1,
                 engine='cudnn', input_shape=None):
        """Local response normalization.

        Args:
            size (int): # of channels to be crossed
                normalization.
            mode (string): 'cross_channel'
            input_shape (tuple): 3d tuple, (channel, height, width)
        """
        conf = self.conf.lrn_conf()
        conf.local_size = size
        conf.alpha = alpha
        conf.beta = beta
        conf.k = k
        # TODO(wangwei) enable mode = 'within_channel'
        assert mode is 'cross_channel', 'only support mode="across_channel"'
        conf.norm_region = NormRegion_ACROSS_CHANNEL
        _check_engine(engine, ['cudnn'])
        self.layer = _create_layer(engine, 'LRN')
        if input_shape is not None:
            self.setup(input_shape)


class Dense(Layer):
    def __init__(self, num_output, use_bias=True, W_specs=None, b_specs=None,
                 W_transpose=True, engine='cuda', name=None, num_input=None):
        """Apply linear/affine transformation, also called inner-product or
        fully connected layer.

        Args:
            num_output (int): output feature length.
            use_bias (bool): add a bias vector or not to the transformed feature
            W_specs (dict): specs for the weight matrix
                'name' for parameter name
                'lr_mult' for learning rate multiplier
                'decay_mult' for weight decay multiplier
                'init' for init method, which could be 'gaussian', 'uniform',
                'xavier' and ''
                'std', 'mean', 'high', 'low' for corresponding init methods
                'clamp' for gradient constraint, value is scalar
                'regularizer' for regularization, currently support 'l2'
            b_specs (dict): specs for the bias vector, same fields as W_specs.
            W_transpose (bool): if true, output=x*W.T+b;
            engine (string): could be 'cudnn', 'cuda'
            num_input (int): input feature length
        """
        super(Dense, Layer).__init__(name)
        conf = self.conf.dense_conf()
        conf.num_output = num_output
        conf.bias_term = use_bias
        conf.transpose = W_transpose
        if W_specs is not None:
            self.conf.add_param(_construct_param_specs_from_dict(W_specs))
        if b_specs is not None:
            self.conf.add_param(_construct_param_specs_from_dict(b_specs))
        if engine is 'cudnn':
            engine = 'cuda'
        _check_engine(engine, 'cudnn', 'cpp')
        self.layer = _create_layer(engine, 'Dense')
        if num_input is not None:
            self.setup((num_input,))


class Dropout(Layer):
    def __init__(self, p=0.5, engine='cudnn', name=None, input_shape=None):
        """Droput layer.

        Args:
            p (float): probability for dropping out the element, i.e., set to 0
            engine (string): 'cudnn' for cudnn version>=5; or 'cuda'
            name (string): layer name
        """
        super(Dropout, Layer).__init__(name)
        conf = self.conf.dropout_conf()
        conf.dropout_ratio = p
        if engine.tolower() is 'cudnn':
            engine = 'cuda'
        _check_engine(engine, ['cuda', 'cpp'])
        self.layer = _create_layer(engine, 'Dropout')
        if input_shape is not None:
            self.setup(input_shape)


class Activation(Layer):
    def __init__(self, mode='relu', engine='cudnn', name=None,
                 input_shape=None):
        """Activation layers.

        Args:
            mode (string): 'relu', 'sigmoid', or 'tanh'
            engine (string): 'cudnn'
            name (string): layer name
            input_shape (tuple): shape of a single sample
        """
        super(Activation, Layer).__init__(name)
        if engine.tolower() is 'cudnn':
            engine = 'cuda'
        _check_engine(engine, ['cuda', 'cpp'])
        mode_dict = {'relu': 'ReLU', 'sigmoid': 'Sigmoid', 'tanh': 'Tanh'}
        self.layer = _create_layer(engine, mode_dict[mode.tolower()])
        if input_shape is not None:
            self.setup(input_shape)


class Softmax(Layer):
    def __init__(self, axis=1, engine='cudnn', name=None, input_shape=None):
        """Apply softmax.

        Args:
            axis (int): reshape the input as a matrix with the dimension
                [0,axis) as the row, the [axis, -1) as the column.
            input_shape (tuple): shape of a single sample
        """
        super(Softmax, Layer).__init__(name)
        conf = self.conf.softmax_conf()
        conf.axis = axis
        self.layer = _create_layer(engine, 'softmax')
        if engine.tolower() is 'cudnn':
            engine = 'cuda'
        _check_engine(engine, ['cuda', 'cpp'])
        self.layer = _create_layer(engine, 'Softmax')
        if input_shape is not None:
            self.setup(input_shape)


class Flatten(Layer):
    def __init__(self, axis=1, engine='cudnn', name=None, input_shape=None):
        """Reshape the input tensor into a matrix.
        Args:
            axis (int): reshape the input as a matrix with the dimension
                [0,axis) as the row, the [axis, -1) as the column.
            input_shape (tuple): shape for a single sample
        """
        super(Flatten, Layer).__init__(name)
        conf = self.conf.flatten_conf()
        conf.axis = axis
        if engine.tolower() is 'cudnn':
            engine = 'cuda'
        _check_engine(engine, ['cuda', 'cpp'])
        self.layer = _create_layer(engine, 'Flatten')
        if input_shape is not None:
            self.setup(input_shape)


def _check_engine(engine, allowed_engines):
    assert engine.tolower() in Set(allowed_engines), \
           '%s is not a supported engine. Pls use one of %s' % \
           (engine, ', '.join(allowed_engines))


def _create_layer(engine, layer):
    layer_type = engine.title() + layer
    # TODO(wangwei) create the swig layer


def _set_kernel_stride_pad(conf, kernel, stride, border_mode, pad):
    """Private function called by Convolution2D and Pooling2D."""
    if type(kernel) is tuple:
        conf.kernel_h = kernel[0]
        conf.kernel_w = kernel[1]
    else:
        conf.kernel = kernel
    if type(stride) is tuple:
        conf.stride_h = stride[0]
        conf.stride_w = stride[1]
    else:
        conf.stride = stride
    mode = border_mode.lower()
    if pad is None:
        # TODO(wangwei) check the border mode
        if mode is 'same':
            pad = (kernel[0] / 2, kernel[1] / 2)
        elif pad is 'valid':
            pad = (0, 0)
        else:
            assert False, ('Unsupported border_mode :%s. '
                           'Please use {"valid", "same"}' % border_mode)
    else:
        assert isinstance(pad, tuple), 'pad should be a tuple'
    conf.pad_h = pad[0]
    conf.pad_w = pad[1]
    return conf


def _construct_param_specs_from_dict(specs):
    """Conver the param specs from a dict into ParamSpec protobuf object.

    Args:
        specs (dict): the fields inlcude
            'name' for parameter name
            'lr_mult' for learning rate multiplier;
            'decay_mult' for weight decay multiplier;
            'init' for init method, which could be 'gaussian', 'uniform',
            'xavier' and '';
            'std', 'mean', 'high', 'low' are used by corresponding init methods;
            'clamp' for gradient constraint, value is scalar.
            'regularizer' for regularization, currently support 'l2'.

    Returns:
        a ParamSpec object
    """
    pass
