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
""" Python layers which wraps the C++ layers by providing easy to construct APIs
"""

from sets import Set
from . import singa_wrap
from .proto import model_pb2
import tensor


class Layer(object):
    """Base Python layer class.

    Usages:
        1.  construct layer without input_sample_shapes, goto 2;
            construct layer with input_sample_shapes, goto 3;
        2. call setup to create the parameters and setup other meta fields
        3. call forward or access layer members
        4. call backward and get parameters for update
    """

    def __init__(self, name, **kwargs):
        self.layer = None  # layer converted by swig
        self.name = name  # TODO(wangwei) duplicate with self.conf.name
        self.conf = model_pb2.LayerConf()
        self.conf.name = name
        self.param_specs = []
        self.has_setup = False

    def param_names(self):
        names = []
        for x in self.param_specs:
            names.append(x['name'])
        return names

    def setup(self, in_shapes):
        """Call the C++ setup function to create params and set some meta data.

        Args:
            in_shapes: if the layer accepts a single input Tensor, in_shapes is
                a single tuple specifying the inpute Tensor shape; if the layer
                accepts multiple input Tensor (e.g., the concatenation layer),
                in_shapes is a tuple of tuples, each for one input Tensor shape
        """
        if self.has_setup:
            return
        self.layer.Setup(list(in_shapes),
                         self.conf.SerializeToString())
        self.has_setup = True

    def get_output_sample_shape(self):
        assert self.has_setup, \
            'Must call setup() before get_output_sample_shape()'
        return self.layer.GetOutputSampleShape()

    def param_values(self):
        """Return param value tensors.

        Do not store these tensors as layer members because cpp Tensor could be
        moved onto diff devices due to the change of layer device. However, the
        py tensors would not update its internal cpp tensor automatically.
        """
        return tensor.from_raw_tensors(self.layer.param_values())

    def forward(self, flag, input):
        assert self.has_setup, 'Must call setup() before forward()'
        assert isinstance(input, tensor.Tensor), 'input must be py Tensor'
        y = self.layer.Forward(flag, input.singa_tensor)
        return tensor.from_raw_tensor(y)

    def backward(self, flag, grad):
        assert isinstance(grad, tensor.Tensor), 'grad must be py Tensor'
        ret = self.layer.Backward(flag, grad.singa_tensor)
        return tensor.from_raw_tensor(ret[0]), tensor.from_raw_tensors(ret[1])

    def to_device(self, device):
        self.layer.ToDevice(device)

    def as_type(self, dtype):
        self.layer.AsType(dtype)

    def __copy__(self):
        pass

    def __deepcopy__(self):
        pass


class Conv2D(Layer):

    def __init__(self, name, nb_kernels, kernel=3, stride=1, border_mode='same',
                 engine='cudnn', cudnn_prefer='fatest', data_format='NCHW',
                 use_bias=True, W_specs=None, b_specs=None,
                 pad=None, input_sample_shape=None):
        """Construct a layer for 2D convolution.

        Args:
            nb_kernels (int): num of the channels (kernels) of the input Tensor
            kernel: an integer or a pair of integers for kernel height and width
            stride: an integer or a pair of integers for stride height and width
            border_mode (string): padding mode, case in-sensitive,
                'valid' -> padding is 0 for height and width
                'same' -> padding is half of the kernel (floor),
                    the kernel must be odd number.
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
                TODO(wangwei) 'clamp' for gradient constraint, value is scalar
                'regularizer' for regularization, currently support 'l2'
            b_specs (dict): hyper-parameters for bias vector, similar as W_specs
            name (string): layer name.
            input_sample_shape: 3d tuple for the shape of the input Tensor
                without the batchsize, e.g., (channel, height, width) or
                (height, width, channel)
        """
        super(Conv2D, self).__init__(name)
        assert data_format == 'NCHW', 'Not supported data format: %s ' \
            'only "NCHW" is enabled currently' % (data_format)
        conf = self.conf.convolution_conf
        conf.num_output = nb_kernels
        conf = _set_kernel_stride_pad(conf, kernel, stride, border_mode, pad)
        conf.bias_term = use_bias
        # TODO(wangwei) enable data format for cpp code
        # conf.data_format = data_format
        if W_specs is None:
            W_specs = {'init': 'xavier'}
        if b_specs is None:
            b_specs = {'init': 'constant'}
        if 'name' not in W_specs:
            W_specs['name'] = name + '_weight'
        if 'name' not in b_specs:
            b_specs['name'] = name + '_bias'
        wspecs = _construct_param_specs_from_dict(W_specs)
        self.conf.param.extend([wspecs])
        self.param_specs.append(wspecs)
        bspecs = _construct_param_specs_from_dict(b_specs)
        self.conf.param.extend([bspecs])
        self.param_specs.append(bspecs)

        _check_engine(engine, ['cudnn'])
        self.layer = _create_layer(engine, 'Convolution')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class Conv1D(Conv2D):

    def __init__(self, name, nb_kernels, kernel=3, stride=1,
                 border_mode='same', engine='cudnn', cudnn_prefer='fatest',
                 use_bias=True, W_specs={'init': 'Xavier'},
                 b_specs={'init': 'Constant', 'value': 0}, pad=None,
                 input_sample_shape=None):
        """Construct a layer for 1D convolution.

        Most of the args are the same as those for Conv2D except the kernel,
        stride, pad, which is a scalar instead of a tuple.
        input_sample_shape is a tuple with a single value for the input feature
        length
        """
        pad = None
        if pad is not None:
            pad = (0, pad)
        if input_sample_shape is not None:
            input_sample_shape = (1, 1, input_sample_shape[0])
        super(Conv1D, self).__init__(name, nb_kernels, (1, kernel), (0, stride),
                                     border_mode, engine, cudnn_prefer,
                                     use_bias=use_bias, pad=pad,
                                     W_specs=W_specs, b_specs=b_specs,
                                     input_sample_shape=input_sample_shape)

    def get_output_sample_shape(self):
        shape = self.layer.GetOutputSampleShape()
        assert len(shape) == 3, 'The output sample shape should be 3D.'\
            'But the length is %d' % len(shape)
        return (shape[0], shape[2])


class Pooling2D(Layer):

    def __init__(self, name, mode, kernel=3, stride=2, border_mode='same',
                 pad=None, data_format='NCHW', engine='cudnn',
                 input_sample_shape=None):
        super(Pooling2D, self).__init__(name)
        assert data_format == 'NCHW', 'Not supported data format: %s ' \
            'only "NCHW" is enabled currently' % (data_format)
        conf = self.conf.pooling_conf
        conf = _set_kernel_stride_pad(conf, kernel, stride, border_mode, pad)
        conf.pool = mode
        _check_engine(engine, ['cudnn'])
        self.layer = _create_layer(engine, 'Pooling')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class MaxPooling2D(Pooling2D):

    def __init__(self, name, kernel=3, stride=2, border_mode='same', pad=None,
                 data_format='NCHW', engine='cudnn', input_sample_shape=None):
        super(MaxPooling2D, self).__init__(name, model_pb2.PoolingConf.MAX,
                                           kernel, stride, border_mode,
                                           pad, data_format, engine,
                                           input_sample_shape)


class AvgPooling2D(Pooling2D):

    def __init__(self, name, kernel=3, stride=2, border_mode='same', pad=None,
                 data_format='NCHW', engine='cudnn', input_sample_shape=None):
        super(AvgPooling2D, self).__init__(name, model_pb2.PoolingConf.AVE,
                                           kernel, stride, border_mode,
                                           pad, data_format, engine,
                                           input_sample_shape)


class MaxPooling1D(MaxPooling2D):

    def __init__(self, name, kernel=3, stride=2, border_mode='same', pad=None,
                 data_format='NCHW', engine='cudnn', input_sample_shape=None):
        """Max pooling for 1D feature.

        Args:
            input_sample_shape (tuple): 1D tuple for input feature length
        """
        pad = None
        if pad is not None:
            pad = (0, pad)
        if input_sample_shape is not None:
            assert len(input_sample_shape) == 1, \
                'AvgPooling1D expects input sample to be 1D'
            input_sample_shape = (1, 1, input_sample_shape[0])
        else:
            input_sample_shape = None
        super(MaxPooling1D, self).__init__(name, (1, kernel), (0, stride),
                                           border_mode, pad,
                                           data_format, engine,
                                           input_sample_shape)

    def get_output_sample_shape(self):
        shape = self.layer.GetOutputSampleShape()
        return (shape[2],)


class AvgPooling1D(AvgPooling2D):

    def __init__(self, name, kernel=3, stride=2, border_mode='same', pad=None,
                 data_format='NCHW', engine='cudnn', input_sample_shape=None):
        """input_feature_length is a scalar value"""
        pad2 = None
        if pad is not None:
            pad2 = (pad, 0)
        if input_sample_shape is not None:
            assert len(input_sample_shape) == 1, \
                'AvgPooling1D expects input sample to be 1D'
            input_sample_shape = (1, 1, input_sample_shape[0])
        else:
            input_sample_shape = None

        super(AvgPooling1D, self).__init__(name, (kernel, 1), (0, stride),
                                           border_mode, pad2,
                                           data_format, engine,
                                           input_sample_shape)

    def get_output_sample_shape(self):
        shape = self.layer.GetOutputSampleShape()
        return (shape[2],)


class BatchNormalization(Layer):
    # TODO(wangwei) add mode and epsilon arguments

    def __init__(self, name, momentum=0.9, engine='cudnn',
                 beta_specs=None, gamma_specs=None, input_sample_shape=None):
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
            input_sample_shape (tuple): with at least one integer
        """
        super(BatchNormalization, self).__init__(name)
        conf = self.conf.batchnorm_conf
        conf.factor = momentum
        if beta_specs is None:
            beta_specs = {'init': 'Xavier'}
        if gamma_specs is None:
            gamma_specs = {'init': 'Xavier'}
        if 'name' not in beta_specs:
            beta_specs['name'] = name + '_beta'
        if 'name' not in gamma_specs:
            gamma_specs['name'] = name + '_gamma'
        self.conf.param.extend([_construct_param_specs_from_dict(beta_specs)])
        self.conf.param.extend([_construct_param_specs_from_dict(gamma_specs)])
        self.param_specs.append(_construct_param_specs_from_dict(beta_specs))
        self.param_specs.append(_construct_param_specs_from_dict(gamma_specs))
        _check_engine(engine, ['cudnn'])
        self.layer = _create_layer(engine, 'BatchNorm')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class LRN(Layer):

    def __init__(self, name, size=5, alpha=1, beta=0.75, mode='cross_channel',
                 k=1, engine='cudnn', input_sample_shape=None):
        """Local response normalization.

        Args:
            size (int): # of channels to be crossed
                normalization.
            mode (string): 'cross_channel'
            input_sample_shape (tuple): 3d tuple, (channel, height, width)
        """
        super(LRN, self).__init__(name)
        conf = self.conf.lrn_conf
        conf.local_size = size
        conf.alpha = alpha
        conf.beta = beta
        conf.k = k
        # TODO(wangwei) enable mode = 'within_channel'
        assert mode == 'cross_channel', 'only support mode="across_channel"'
        conf.norm_region = model_pb2.LRNConf.ACROSS_CHANNELS
        _check_engine(engine, ['cudnn'])
        self.layer = _create_layer(engine, 'LRN')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class Dense(Layer):

    def __init__(self, name, num_output, use_bias=True,
                 W_specs=None, b_specs=None,
                 W_transpose=True, engine='cuda', input_sample_shape=None):
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
            input_sample_shape (tuple): input feature length
        """
        super(Dense, self).__init__(name)
        conf = self.conf.dense_conf
        conf.num_output = num_output
        conf.bias_term = use_bias
        conf.transpose = W_transpose
        if W_specs is None:
            W_specs = {'init': 'xavier'}
        if b_specs is None:
            b_specs = {'init': 'constant'}
        if 'name' not in W_specs:
            W_specs['name'] = name + '_weight'
        if 'name' not in b_specs:
            b_specs['name'] = name + '_bias'
        self.conf.param.extend([_construct_param_specs_from_dict(W_specs)])
        self.param_specs.append(_construct_param_specs_from_dict(W_specs))
        self.conf.param.extend([_construct_param_specs_from_dict(b_specs)])
        self.param_specs.append(_construct_param_specs_from_dict(b_specs))
        if engine == 'cudnn':
            engine = 'cuda'
        _check_engine(engine, ['cuda', 'cpp'])
        self.layer = _create_layer(engine, 'Dense')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class Dropout(Layer):

    def __init__(self, name, p=0.5, engine='cuda', input_sample_shape=None):
        """Droput layer.

        Args:
            p (float): probability for dropping out the element, i.e., set to 0
            engine (string): 'cudnn' for cudnn version>=5; or 'cuda'
            name (string): layer name
        """
        super(Dropout, self).__init__(name)
        conf = self.conf.dropout_conf
        conf.dropout_ratio = p
        # 'cudnn' works for v>=5.0
        #  if engine.lower() == 'cudnn':
        #      engine = 'cuda'
        _check_engine(engine, ['cudnn', 'cuda', 'cpp'])
        self.layer = _create_layer(engine, 'Dropout')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class Activation(Layer):

    def __init__(self, name, mode='relu', engine='cudnn',
                 input_sample_shape=None):
        """Activation layers.

        Args:
            engine (string): 'cudnn'
            name (string): layer name
            mode (string): 'relu', 'sigmoid', or 'tanh'
            input_sample_shape (tuple): shape of a single sample
        """
        super(Activation, self).__init__(name)
        _check_engine(engine, ['cudnn', 'cuda', 'cpp'])
        mode_dict = {'relu': 'RELU', 'sigmoid': 'SIGMOID', 'tanh': 'TANH'}
        self.conf.type = mode_dict[mode.lower()]
        self.layer = _create_layer(engine, 'Activation')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class Softmax(Layer):

    def __init__(self, name, axis=1, engine='cudnn', input_sample_shape=None):
        """Apply softmax.

        Args:
            axis (int): reshape the input as a matrix with the dimension
                [0,axis) as the row, the [axis, -1) as the column.
            input_sample_shape (tuple): shape of a single sample
        """
        super(Softmax, self).__init__(name)
        # conf = self.conf.softmax_conf
        # conf.axis = axis
        _check_engine(engine, ['cudnn', 'cuda', 'cpp'])
        self.layer = _create_layer(engine, 'Softmax')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class Flatten(Layer):

    def __init__(self, name, axis=1, engine='cudnn', input_sample_shape=None):
        """Reshape the input tensor into a matrix.
        Args:
            axis (int): reshape the input as a matrix with the dimension
                [0,axis) as the row, the [axis, -1) as the column.
            input_sample_shape (tuple): shape for a single sample
        """
        super(Flatten, self).__init__(name)
        conf = self.conf.flatten_conf
        conf.axis = axis
        _check_engine(engine, ['cudnn', 'cuda', 'cpp'])
        if engine == 'cudnn':
            engine = 'cuda'
        self.layer = _create_layer(engine, 'Flatten')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


def _check_engine(engine, allowed_engines):
    assert engine.lower() in Set(allowed_engines), \
           '%s is not a supported engine. Pls use one of %s' % \
           (engine, ', '.join(allowed_engines))


def _create_layer(engine, layer):
    if engine == 'cuda' or engine == 'cpp':
        layer_type = layer
    else:
        layer_type = engine.title() + layer
    return singa_wrap.CreateLayer(layer_type)


def _set_kernel_stride_pad(conf, kernel, stride, border_mode, pad):
    """Private function called by Convolution2D and Pooling2D."""
    if isinstance(kernel, tuple):
        conf.kernel_h = kernel[0]
        conf.kernel_w = kernel[1]
    else:
        conf.kernel_h = kernel
        conf.kernel_w = kernel
    if isinstance(stride, tuple):
        conf.stride_h = stride[0]
        conf.stride_w = stride[1]
    else:
        conf.stride_h = stride
        conf.stride_w = stride
    mode = border_mode.lower()
    if pad is None:
        # TODO(wangwei) check the border mode
        if mode == 'same':
            assert conf.kernel_h % 2 == 1 and conf.kernel_w % 2 == 1, \
                'Must use odd kernel for mode="same", kernel is (%d, %d)' % (
                    conf.kernel_h, conf.kernel_w)
            pad = (conf.kernel_h / 2, conf.kernel_w / 2)
        elif mode == 'valid':
            pad = (0, 0)
        else:
            assert False, ('Unsupported border_mode: %s. '
                           'Please use {"valid", "same"}' % border_mode)
        assert isinstance(pad, tuple), 'pad should be a tuple'
    if isinstance(pad, tuple):
        conf.pad_h = pad[0]
        conf.pad_w = pad[1]
    else:
        conf.pad_h = pad
        conf.pad_w = pad
    return conf


def _construct_param_specs_from_dict(specs):
    """Conver the param specs from a dict into ParamSpec protobuf object.

    Args:
        specs (dict): the fields inlcude
            'name' for parameter name
            'lr_mult' for learning rate multiplier;
            'decay_mult' for weight decay multiplier;
            'init' for init method, which could be 'gaussian', 'uniform',
            'xavier' and 'msra';
            'std', 'mean', 'high', 'low' are used by corresponding init methods;
            'constraint' for gradient constraint, value is a float threshold for
                clampping the gradient.
            'regularizer' for regularization, currently support 'l2', value is a
                float for the coefficient.

    Returns:
        a ParamSpec object
    """
    conf = model_pb2.ParamSpec()
    if 'name' in specs:
        conf.name = specs['name']
    if 'lr_mult' in specs:
        conf.lr_mult = specs['lr_mult']
    if 'decay_mult' in specs:
        conf.decay_mult = specs['decay_mult']
    if 'init' in specs:
        filler = conf.filler
        filler.type = specs['init'].lower()
        if specs['init'].lower() == 'uniform':
            assert 'low' in specs and 'high' in specs, \
                'low and high are required for "uniform" init method'
            filler.low = specs['low']
            filler.high = specs['high']
        elif specs['init'].lower() == 'gaussian':
            assert 'mean' in specs and 'std' in specs, \
                'std and mean are required for "gaussian" init method'
            filler.mean = specs['mean']
            filler.std = specs['std']
        elif specs['init'].lower() == 'constant' and 'value' in specs:
            filler.value = specs['value']
    if 'regularizer' in specs:
        conf.regularizer.coefficient = specs['regularizer']
    if 'constraint' in specs:
        conf.constraint.threshold = specs['constraint']
    return conf


def get_layer_list():
    """ Return a list of strings reprensenting the all supported layers"""
    return singa_wrap.GetRegisteredLayers()
