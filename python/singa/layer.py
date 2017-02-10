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
""" Python layers wrap the C++ layers to provide simpler construction APIs.

Example usages::

    from singa import layer
    from singa import tensor
    from singa import device

    layer.engine = 'cudnn'  # to use cudnn layers
    dev = device.create_cuda_gpu()

    # create a convolution layer
    conv = layer.Conv2D('conv', 32, 3, 1, pad=1, input_sample_shape=(3, 32, 32))
    conv.to_device(dev)  # move the layer data onto a CudaGPU device
    x = tensor.Tensor((3, 32, 32), dev)
    x.uniform(-1, 1)
    y = conv.foward(True, x)

    dy = tensor.Tensor()
    dy.reset_like(y)
    dy.set_value(0.1)
    # dp is a list of tensors for parameter gradients
    dx, dp = conv.backward(kTrain, dy)
"""

from sets import Set
from . import singa_wrap
from .proto import model_pb2
import tensor


engine = 'cudnn'
'''engine is the prefix of layer identifier.

The value could be one of [**'cudnn', 'singacpp', 'singacuda', 'singacl'**], for
layers implemented using the cudnn library, Cpp, Cuda and OpenCL respectively.
For example, CudnnConvolution layer is identified by 'cudnn_convolution';
'singacpp_convolution' is for Convolution layer;
Some layers' implementation use only Tensor functions, thererfore they are
transparent to the underlying devices. For threse layers, they would have
multiple identifiers, e.g., singacpp_dropout, singacuda_dropout and
singacl_dropout are all for the Dropout layer. In addition, it has an extra
identifier 'singa', i.e. 'singa_dropout' also stands for the Dropout layer.

engine is case insensitive. Each python layer would create the correct specific
layer using the engine attribute.
'''

if singa_wrap.USE_CUDNN:
    cudnn_version = singa_wrap.CUDNN_VERSION
else:
    cudnn_version = 0


class Layer(object):
    '''Base Python layer class.

    Typically, the life cycle of a layer instance includes:
        1. construct layer without input_sample_shapes, goto 2;
           construct layer with input_sample_shapes, goto 3;
        2. call setup to create the parameters and setup other meta fields
        3. call forward or access layer members
        4. call backward and get parameters for update

    Args:
        name (str): layer name
    '''
    def __init__(self, name, conf=None, **kwargs):
        if conf is None:
            self.layer = None  # layer converted by swig
            self.name = name  # TODO(wangwei) duplicate with self.conf.name
            self.conf = model_pb2.LayerConf()
            self.conf.name = name
            self.param_specs = []
        else:
            self.conf = conf
            self.name = str(conf.name)
            self.caffe_layer()
            self.param_specs = []

            # convert caffe proto into singa proto format
            #   case1: parameters of conv and dense layers
            #   case2: type of activation layers
            if (conf.type == 'Convolution' or conf.type == 4) or \
                    (conf.type == 'InnerProduct' or conf.type == 14):
                w, b = _construct_param_specs_from_caffe_proto(conf)
                del conf.param[:]
                conf.param.extend([w, b])
                self.param_specs.append(w)
                self.param_specs.append(b)
                # print 'conf:\n', conf
            if conf.type == 'Pooling':
                conf.pooling_conf.ceil = True
                # print 'conf:\n', conf
            elif (conf.type == 'ReLU' or conf.type == 18 or
                  conf.type == 'Sigmoid' or conf.type == 19 or
                  conf.type == 'TanH' or conf.type == 23):
                conf.type = (engine + '_' + conf.type).lower()
            self.conf = conf

        self.has_setup = False

    def setup(self, in_shapes):
        '''Call the C++ setup function to create params and set some meta data.

        Args:
            in_shapes: if the layer accepts a single input Tensor, in_shapes is
                a single tuple specifying the inpute Tensor shape; if the layer
                accepts multiple input Tensor (e.g., the concatenation layer),
                in_shapes is a tuple of tuples, each for one input Tensor
        '''
        if self.has_setup:
            return
        if type(in_shapes[0]) is tuple:
            self.layer.SetupWithMultInputs([list(s) for s in in_shapes],
                                           self.conf.SerializeToString())
        else:
            self.layer.Setup(list(in_shapes), self.conf.SerializeToString())
        self.has_setup = True

    def caffe_layer(self):
        '''
        Create a singa layer based on caffe layer configuration.
        '''
        _check_engine(engine, ['cudnn', 'singacpp', 'singacuda', 'singacl'])
        if self.conf.type == 'InnerProduct' or self.conf.type == 14:
            self.layer = _create_layer(engine, 'Dense')
        else:
            self.layer = _create_layer(engine, str(self.conf.type))

    def get_output_sample_shape(self):
        '''Called after setup to get the shape of the output sample(s).

        Returns:
            a tuple for a single output Tensor or a list of tuples if this layer
            has multiple outputs
        '''
        assert self.has_setup, \
            'Must call setup() before get_output_sample_shape()'
        return self.layer.GetOutputSampleShape()

    def param_names(self):
        '''
        Returns:
            a list of strings, one for the name of one parameter Tensor
        '''
        names = []
        for x in self.param_specs:
            names.append(x['name'])
        return names

    def param_values(self):
        '''Return param value tensors.

        Parameter tensors are not stored as layer members because cpp Tensor
        could be moved onto diff devices due to the change of layer device,
        which would result in inconsistency.

        Returns:
            a list of tensors, one for each paramter
        '''
        if self.layer is None:
            return []
        else:
            return tensor.from_raw_tensors(self.layer.param_values())

    def forward(self, flag, x):
        '''Forward propagate through this layer.

        Args:
            flag: True (kTrain) for training (kEval); False for evaluating;
                other values for furture use.
            x (Tensor or list<Tensor>): an input tensor if the layer is
                connected from a single layer; a list of tensors if the layer
                is connected from multiple layers.

        Return:
            a tensor if the layer is connected to a single layer; a list of
            tensors if the layer is connected to multiple layers;
        '''
        assert self.has_setup, 'Must call setup() before forward()'
        if type(flag) is bool:
            if flag:
                flag = model_pb2.kTrain
            else:
                flag = model_pb2.kEval
        if type(x) is list:
            xs = [t.singa_tensor for t in x]
            y = self.layer.ForwardWithMultInputs(flag, xs)
        else:
            assert isinstance(x, tensor.Tensor), \
                    'input of %s (type:%s) must be a Tensor or Tensor list'\
                    % (self.name, type(x).__name__)
            y = self.layer.Forward(flag, x.singa_tensor)
        if type(y) is tuple:
            return tensor.from_raw_tensors(y)
        else:
            return tensor.from_raw_tensor(y)

    def backward(self, flag, dy):
        '''Backward propagate gradients through this layer.

        Args:
            flag (int): for future use.
            dy (Tensor or list<Tensor>): the gradient tensor(s) y w.r.t the
                objective loss
        Return:
            <dx, <dp1, dp2..>>, dx is a (set of) tensor(s) for the gradient of x
            , dpi is the gradient of the i-th parameter
        '''
        if type(flag) is bool:
            if flag:
                flag = model_pb2.kTrain
            else:
                flag = model_pb2.kEval

        if type(dy) == list:
            dys = [t.singa_tensor for t in dy]
            ret = self.layer.BackwardWithMultInputs(flag, dys)
        else:
            assert isinstance(dy, tensor.Tensor), \
                    'input of %s (type:%s) must be a Tensor or Tensor list'\
                    % (self.name, type(dy).__name__)
            dys = dy.singa_tensor
            ret = self.layer.Backward(flag, dys)
        if type(ret[0]) is tuple:
            dxs = tensor.from_raw_tensors(ret[0])
        else:
            dxs = tensor.from_raw_tensor(ret[0])
        return dxs, tensor.from_raw_tensors(ret[1])

    def to_device(self, device):
        '''Move layer state tensors onto the given device.

        Args:
            device: swig converted device, created using singa.device
        '''
        if self.layer is not None:
            self.layer.ToDevice(device)

    def as_type(self, dtype):
        pass

    def __copy__(self):
        pass

    def __deepcopy__(self):
        pass


class Dummy(Layer):
    '''A dummy layer that does nothing but just forwards/backwards the data
    (the input/output is a single tensor).
    '''
    def __init__(self, name, input_sample_shape=None):
        super(Dummy, self).__init__(name)
        self.output_sample_shape = input_sample_shape

    def get_output_sample_shape(self):
        return self.output_sample_shape

    def setup(self, input_sample_shape):
        self.output_sample_shape = input_sample_shape
        self.has_setup = True

    def forward(self, flag, x):
        '''Return the input x'''
        return x

    def backward(self, falg, dy):
        '''Return dy, []'''
        return dy, []


class Conv2D(Layer):
    """Construct a layer for 2D convolution.

    Args:
        nb_kernels (int): num of the channels (kernels) of the input Tensor
        kernel: an integer or a pair of integers for kernel height and width
        stride: an integer or a pair of integers for stride height and width
        border_mode (string): padding mode, case in-sensitive,
            'valid' -> padding is 0 for height and width
            'same' -> padding is half of the kernel (floor), the kernel must be
            odd number.
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

    def __init__(self, name, nb_kernels, kernel=3, stride=1, border_mode='same',
                 cudnn_prefer='fatest', data_format='NCHW',
                 use_bias=True, W_specs=None, b_specs=None,
                 pad=None, input_sample_shape=None):
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
        if 'name' not in W_specs:
            W_specs['name'] = name + '_weight'
        wspecs = _construct_param_specs_from_dict(W_specs)
        self.conf.param.extend([wspecs])
        self.param_specs.append(wspecs)
        if use_bias:
            if b_specs is None:
                b_specs = {'init': 'constant'}
            if 'name' not in b_specs:
                b_specs['name'] = name + '_bias'
            bspecs = _construct_param_specs_from_dict(b_specs)
            self.conf.param.extend([bspecs])
            self.param_specs.append(bspecs)

        _check_engine(engine, ['cudnn', 'singacpp', 'singacl'])
        self.layer = _create_layer(engine, 'Convolution')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class Conv1D(Conv2D):
    """Construct a layer for 1D convolution.

    Most of the args are the same as those for Conv2D except the kernel,
    stride, pad, which is a scalar instead of a tuple.
    input_sample_shape is a tuple with a single value for the input feature
    length
    """

    def __init__(self, name, nb_kernels, kernel=3, stride=1,
                 border_mode='same', cudnn_prefer='fatest',
                 use_bias=True, W_specs={'init': 'Xavier'},
                 b_specs={'init': 'Constant', 'value': 0}, pad=None,
                 input_sample_shape=None):
        pad = None
        if pad is not None:
            pad = (0, pad)
        if input_sample_shape is not None:
            input_sample_shape = (1, 1, input_sample_shape[0])
        super(Conv1D, self).__init__(name, nb_kernels, (1, kernel), (0, stride),
                                     border_mode, cudnn_prefer,
                                     use_bias=use_bias, pad=pad,
                                     W_specs=W_specs, b_specs=b_specs,
                                     input_sample_shape=input_sample_shape)

    def get_output_sample_shape(self):
        shape = self.layer.GetOutputSampleShape()
        assert len(shape) == 3, 'The output sample shape should be 3D.'\
            'But the length is %d' % len(shape)
        return (shape[0], shape[2])


class Pooling2D(Layer):
    '''2D pooling layer providing max/avg pooling.

    All args are the same as those for Conv2D, except the following one

    Args:
        mode: pooling type, model_pb2.PoolingConf.MAX or
            model_pb2.PoolingConf.AVE

    '''

    def __init__(self, name, mode, kernel=3, stride=2, border_mode='same',
                 pad=None, data_format='NCHW', input_sample_shape=None):
        super(Pooling2D, self).__init__(name)
        assert data_format == 'NCHW', 'Not supported data format: %s ' \
            'only "NCHW" is enabled currently' % (data_format)
        conf = self.conf.pooling_conf
        conf = _set_kernel_stride_pad(conf, kernel, stride, border_mode, pad)
        conf.pool = mode
        _check_engine(engine, ['cudnn', 'singacpp', 'singacl'])
        self.layer = _create_layer(engine, 'Pooling')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class MaxPooling2D(Pooling2D):

    def __init__(self, name, kernel=3, stride=2, border_mode='same', pad=None,
                 data_format='NCHW', input_sample_shape=None):
        super(MaxPooling2D, self).__init__(name, model_pb2.PoolingConf.MAX,
                                           kernel, stride, border_mode,
                                           pad, data_format, input_sample_shape)


class AvgPooling2D(Pooling2D):

    def __init__(self, name, kernel=3, stride=2, border_mode='same', pad=None,
                 data_format='NCHW', input_sample_shape=None):
        super(AvgPooling2D, self).__init__(name, model_pb2.PoolingConf.AVE,
                                           kernel, stride, border_mode,
                                           pad, data_format, input_sample_shape)


class MaxPooling1D(MaxPooling2D):

    def __init__(self, name, kernel=3, stride=2, border_mode='same', pad=None,
                 data_format='NCHW', input_sample_shape=None):
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
                                           data_format, input_sample_shape)

    def get_output_sample_shape(self):
        shape = self.layer.GetOutputSampleShape()
        return (shape[2],)


class AvgPooling1D(AvgPooling2D):

    def __init__(self, name, kernel=3, stride=2, border_mode='same', pad=None,
                 data_format='NCHW', input_sample_shape=None):
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
                                           data_format, input_sample_shape)

    def get_output_sample_shape(self):
        shape = self.layer.GetOutputSampleShape()
        return (shape[2],)


class BatchNormalization(Layer):
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

    def __init__(self, name, momentum=0.9,
                 beta_specs=None, gamma_specs=None, input_sample_shape=None):
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
        mean_specs = {'init': 'constant', 'value': 0, 'name': name + '_mean'}
        var_specs = {'init': 'constant', 'value': 1, 'name': name + '_var'}
        self.conf.param.extend([_construct_param_specs_from_dict(gamma_specs)])
        self.conf.param.extend([_construct_param_specs_from_dict(beta_specs)])
        self.conf.param.extend([_construct_param_specs_from_dict(mean_specs)])
        self.conf.param.extend([_construct_param_specs_from_dict(var_specs)])
        self.param_specs.append(_construct_param_specs_from_dict(gamma_specs))
        self.param_specs.append(_construct_param_specs_from_dict(beta_specs))
        self.param_specs.append(_construct_param_specs_from_dict(mean_specs))
        self.param_specs.append(_construct_param_specs_from_dict(var_specs))
        _check_engine(engine, ['cudnn', 'singa', 'singacpp', 'singacuda',
                               'singacl'])
        self.layer = _create_layer(engine, 'BatchNorm')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class LRN(Layer):
    """Local response normalization.

    Args:
        size (int): # of channels to be crossed
            normalization.
        mode (string): 'cross_channel'
        input_sample_shape (tuple): 3d tuple, (channel, height, width)
    """

    def __init__(self, name, size=5, alpha=1, beta=0.75, mode='cross_channel',
                 k=1, input_sample_shape=None):
        super(LRN, self).__init__(name)
        conf = self.conf.lrn_conf
        conf.local_size = size
        conf.alpha = alpha
        conf.beta = beta
        conf.k = k
        # TODO(wangwei) enable mode = 'within_channel'
        assert mode == 'cross_channel', 'only support mode="across_channel"'
        conf.norm_region = model_pb2.LRNConf.ACROSS_CHANNELS
        _check_engine(engine, ['cudnn', 'singa', 'singacpp', 'singacuda',
                               'singacl'])
        self.layer = _create_layer(engine, 'LRN')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class Dense(Layer):
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
        input_sample_shape (tuple): input feature length
    """

    def __init__(self, name, num_output, use_bias=True,
                 W_specs=None, b_specs=None,
                 W_transpose=False, input_sample_shape=None):
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
            input_sample_shape (tuple): input feature length
        """
        super(Dense, self).__init__(name)
        conf = self.conf.dense_conf
        conf.num_output = num_output
        conf.bias_term = use_bias
        conf.transpose = W_transpose
        if W_specs is None:
            W_specs = {'init': 'xavier'}
        if 'name' not in W_specs:
            W_specs['name'] = name + '_weight'
        wspecs = _construct_param_specs_from_dict(W_specs)
        self.conf.param.extend([wspecs])
        self.param_specs.append(wspecs)
        if use_bias:
            if b_specs is None:
                b_specs = {'init': 'constant', 'value': 0}
            if 'name' not in b_specs:
                b_specs['name'] = name + '_bias'
            bspecs = _construct_param_specs_from_dict(b_specs)
            self.conf.param.extend([bspecs])
            self.param_specs.append(bspecs)
        # dense layer is transparent to engine.
        if engine == 'cudnn':
            self.layer = _create_layer('singacuda', 'Dense')
        else:
            self.layer = _create_layer(engine, 'Dense')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class Dropout(Layer):
    """Droput layer.

    Args:
        p (float): probability for dropping out the element, i.e., set to 0
        name (string): layer name
    """

    def __init__(self, name, p=0.5, input_sample_shape=None):
        super(Dropout, self).__init__(name)
        conf = self.conf.dropout_conf
        conf.dropout_ratio = p
        # dropout is support in cudnn since V5
        if engine.lower() == 'cudnn' and cudnn_version < 5000:
            myengine = 'singacuda'
        else:
            myengine = engine
        _check_engine(myengine, ['cudnn', 'singa', 'singacpp', 'singacuda',
                                 'singacl'])
        self.layer = _create_layer(myengine, 'Dropout')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class Activation(Layer):
    """Activation layers.

    Args:
        name (string): layer name
        mode (string): 'relu', 'sigmoid', or 'tanh'
        input_sample_shape (tuple): shape of a single sample
    """

    def __init__(self, name, mode='relu', input_sample_shape=None):
        super(Activation, self).__init__(name)
        _check_engine(engine, ['cudnn', 'singacpp', 'singacuda', 'singacl'])
        self.conf.type = (engine + '_' + mode).lower()
        self.layer = _create_layer(engine, mode)
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class Softmax(Layer):
    """Apply softmax.

    Args:
        axis (int): reshape the input as a matrix with the dimension
            [0,axis) as the row, the [axis, -1) as the column.
        input_sample_shape (tuple): shape of a single sample
    """

    def __init__(self, name, axis=1, input_sample_shape=None):
        super(Softmax, self).__init__(name)
        # conf = self.conf.softmax_conf
        # conf.axis = axis
        _check_engine(engine, ['cudnn', 'singa', 'singacpp', 'singacl',
                               'singacuda'])
        self.layer = _create_layer(engine, 'Softmax')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class Flatten(Layer):
    """Reshape the input tensor into a matrix.

    Args:
        axis (int): reshape the input as a matrix with the dimension
            [0,axis) as the row, the [axis, -1) as the column.
        input_sample_shape (tuple): shape for a single sample
    """

    def __init__(self, name, axis=1, input_sample_shape=None):
        super(Flatten, self).__init__(name)
        conf = self.conf.flatten_conf
        conf.axis = axis
        # fltten layer is transparent to engine
        if engine == 'cudnn':
            self.layer = _create_layer('singacuda', 'Flatten')
        else:
            self.layer = _create_layer(engine, 'Flatten')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)


class Merge(Layer):
    '''Sum all input tensors.

    Args:
        input_sample_shape: sample shape of the input. The sample shape of all
            inputs should be the same.
    '''

    def __init__(self, name, input_sample_shape=None):
        self.in_shape = input_sample_shape
        self.num_input = 1
        super(Merge, self).__init__(name)

    def setup(self, in_shape):
        self.in_shape = in_shape
        self.has_setup = True

    def get_output_sample_shape(self):
        return self.in_shape

    def forward(self, flag, inputs):
        '''Merge all input tensors by summation.

        TODO(wangwei) do element-wise merge operations, e.g., avg, count
        Args:
            flag: not used.
            inputs (list): a list of tensors

        Returns:
            A single tensor as the sum of all input tensors
        '''
        assert len(inputs) > 1, 'There must be multiple input tensors'
        self.num_input = len(inputs)
        output = tensor.Tensor()
        output.reset_like(inputs[0])
        output.set_value(0)
        for x in inputs:
            output += x
        return output

    def backward(self, flag, grad):
        '''Replicate the grad for each input source layer.

        Args:
            grad(Tensor), the gradient tensor of the merged result from forward

        Returns:
            A list of replicated grad, one per source layer
        '''
        assert isinstance(grad, tensor.Tensor), 'The input must be Tensor'\
                ' instead of %s' % type(grad).__name__
        return [grad] * self.num_input, []  # * self.num_input


class Split(Layer):
    '''Replicate the input tensor.

    Args:
        num_output (int): number of output tensors to generate.
        input_sample_shape: includes a single integer for the input sample
            feature size.
    '''
    def __init__(self, name, num_output, input_sample_shape=None):
        self.num_output = num_output
        self.in_shape = input_sample_shape
        super(Split, self).__init__(name)

    def setup(self, in_shape):
        self.in_shape = in_shape
        self.has_setup = True

    def get_output_sample_shape(self):
        return [self.in_shape] * self.num_output

    def forward(self, flag, input):
        '''Replicate the input tensor into mutiple tensors.

        Args:
            flag: not used
            input: a single input tensor

        Returns:
            a list a output tensor (each one is a copy of the input)
        '''
        assert isinstance(input, tensor.Tensor), 'The input must be Tensor'
        outputs = [input] * self.num_output
        return outputs

    def backward(self, flag, grads):
        '''Sum all grad tensors to generate a single output tensor.

        Args:
            grads(list of Tensor), one per dest layer

        Returns:
            a single tensor as the sum of all grads
        '''
        assert len(grads) > 1, 'There must be multiple gradients'
        dx = tensor.Tensor()
        dx.reset_like(grads[0])
        dx.set_value(0)
        for g in grads:
            dx += g
        return dx, []


class Concat(Layer):
    '''Concatenate tensors vertically (axis = 0) or horizontally (axis = 1).

    Currently, only support tensors with 2 dimensions.

    Args:
        axis(int): 0 for concat row; 1 for concat columns;
        input_sample_shapes: a list of sample shape tuples, one per input tensor
    '''

    def __init__(self, name, axis, input_sample_shapes=None):
        super(Concat, self).__init__(name)
        self.in_shapes = input_sample_shapes
        self.axis = axis
        self.conf.concat_conf.axis = axis
        if engine == "cudnn":
            self.layer = _create_layer('singacuda', 'Concat')
        else:
            self.layer = _create_layer(engine, 'Concat')
        if input_sample_shapes is not None:
            self.setup(input_sample_shapes)

    def forward(self, flag, inputs):
        '''Concatenate all input tensors.

        Args:
            flag: same as Layer::forward()
            input: a list of tensors

        Returns:
            a single concatenated tensor
        '''
        assert type(inputs) is list, 'Must be a list of Tensors'
        ys = super(Concat, self).forward(flag, inputs)
        return ys[0]

    def backward(self, flag, dy):
        '''Backward propagate gradients through this layer.

        Args:
            flag: same as Layer::backward()
            dy(Tensor): the gradient tensors of y w.r.t objective loss
        Return:
            <dx, []>, dx is a list tensors for the gradient of the inputs; []
               is an empty list.
        '''
        if type(dy) is tensor.Tensor:
            dy = [dy]
        assert type(dy) is list, 'Must be a list(Tensor)'
        return super(Concat, self).backward(flag, dy)


class Slice(Layer):
    '''Slice the input tensor into multiple sub-tensors vertially (axis=0) or
    horizontally (axis=1).

    Args:
        axis (int): 0 for slice rows; 1 for slice columns;
        slice_point(list): positions along the axis to do slice; there are n-1
            points for n sub-tensors;
        input_sample_shape: input tensor sample shape
    '''

    def __init__(self, name, axis, slice_point, input_sample_shape=None):
        super(Slice, self).__init__(name)
        self.in_shape = input_sample_shape
        self.axis = axis
        self.conf.slice_conf.axis = axis
        self.conf.slice_conf.slice_point.extend(slice_point)
        if engine == "cudnn":
            self.layer = _create_layer('singacuda', 'Slice')
        else:
            self.layer = _create_layer(engine, 'Slice')
        if input_sample_shape is not None:
            self.setup(input_sample_shape)

    def get_output_sample_shape(self):
        out = []
        for i in range(len(self.conf.slice_conf.slice_point) + 1):
            out.append(self.layer.GetOutputSampleShape(i))

    def forward(self, flag, x):
        '''Slice the input tensor on the given axis.

        Args:
            flag: same as Layer::forward()
            x: a single input tensor

        Returns:
            a list a output tensor
        '''
        if type(x) is tensor.Tensor:
            x = [x]
        assert type(x) is list, 'Must be a list of Tensor'
        return super(Slice, self).forward(flag, x)

    def backward(self, flag, grads):
        '''Concate all grad tensors to generate a single output tensor

        Args:
            flag: same as Layer::backward()
            grads: a list of tensors, one for the gradient of one sliced tensor

        Returns:
            a single tensor for the gradient of the original user, and an empty
                list.
        '''
        assert len(grads) > 1, 'There must be multiple gradients'
        dxs, _ = super(Slice, self).backward(flag, grads)
        return dxs[0], []


class RNN(Layer):
    '''Recurrent layer with 4 types of units, namely lstm, gru, tanh and relu.

    Args:
        hidden_size: hidden feature size, the same for all stacks of layers.
        rnn_mode: decides the rnn unit, which could be one of 'lstm', 'gru',
            'tanh' and 'relu', refer to cudnn manual for each mode.
        num_stacks: num of stacks of rnn layers. It is different to the
            unrolling seqence length.
        input_mode: 'linear' convert the input feature x by by a linear
            transformation to get a feature vector of size hidden_size;
            'skip' does nothing but requires the input feature size equals
            hidden_size
        bidirection: True for bidirectional RNN
        param_specs: config for initializing the RNN parameters.
        input_sample_shape: includes a single integer for the input sample
            feature size.
    '''

    def __init__(self, name, hidden_size, rnn_mode='lstm', dropout=0.0,
                 num_stacks=1, input_mode='linear', bidirectional=False,
                 param_specs=None, input_sample_shape=None):
        assert cudnn_version >= 5005, 'RNN is supported since CUDNN V5.0.5; '\
            'This version is %d' % cudnn_version
        super(RNN, self).__init__(name)
        conf = self.conf.rnn_conf
        assert hidden_size > 0, 'Hidden feature size must > 0'
        conf.hidden_size = hidden_size
        assert rnn_mode in Set(['lstm', 'gru', 'tanh', 'relu']),  \
            'rnn mode %s is not available' % (rnn_mode)
        conf.rnn_mode = rnn_mode
        conf.num_stacks = num_stacks
        conf.dropout = dropout
        conf.input_mode = input_mode
        conf.direction = 'unidirectional'
        if bidirectional:
            conf.direction = 'bidirectional'
        # currently only has rnn layer implemented using cudnn
        _check_engine(engine, ['cudnn'])
        if param_specs is None:
            param_specs = {'name': name + '-weight',
                           'init': 'uniform', 'low': 0, 'high': 1}
        self.conf.param.extend([_construct_param_specs_from_dict(param_specs)])
        self.param_specs.append(_construct_param_specs_from_dict(param_specs))

        self.layer = singa_wrap.CudnnRNN()
        if input_sample_shape is not None:
            self.setup(input_sample_shape)

    def forward(self, flag, inputs):
        '''Forward inputs through the RNN.

        Args:
            flag: True(kTrain) for training; False(kEval) for evaluation;
                others values for future use.
            inputs, <x1, x2,...xn, hx, cx>, where xi is the input tensor for the
                i-th position, its shape is (batch_size, input_feature_length);
                the batch_size of xi must >= that of xi+1; hx is the initial
                hidden state of shape (num_stacks * bidirection?2:1, batch_size,
                hidden_size). cx is the initial cell state tensor of the same
                shape as hy. cx is valid for only lstm. For other RNNs there is
                no cx. Both hx and cx could be dummy tensors without shape and
                data.

        Returns:
            <y1, y2, ... yn, hy, cy>, where yi is the output tensor for the i-th
                position, its shape is (batch_size,
                hidden_size * bidirection?2:1). hy is the final hidden state
                tensor. cx is the final cell state tensor. cx is only used for
                lstm.
        '''
        assert self.has_setup, 'Must call setup() before forward()'
        assert len(inputs) > 1, 'The input to RNN must include at '\
            'least one input tensor '\
            'and one hidden state tensor (could be a dummy tensor)'
        tensors = []
        for t in inputs:
            assert isinstance(t, tensor.Tensor), \
                'input must be py Tensor %s' % (type(t))
            tensors.append(t.singa_tensor)
        if type(flag) is bool:
            if flag:
                flag = model_pb2.kTrain
            else:
                flag = model_pb2.kEval
        y = self.layer.ForwardWithMultInputs(flag, tensors)
        return tensor.from_raw_tensors(y)

    def backward(self, flag, grad):
        '''Backward gradients through the RNN.

        Args:
            flag, for future use.
            grad, <dy1, dy2,...dyn, dhy, dcy>, where dyi is the gradient for the
            i-th output, its shape is (batch_size, hidden_size*bidirection?2:1);
                dhy is the gradient for the final hidden state, its shape is
                (num_stacks * bidirection?2:1, batch_size,
                hidden_size). dcy is the gradient for the final cell state.
                cx is valid only for lstm. For other RNNs there is
                no cx. Both dhy and dcy could be dummy tensors without shape and
                data.

        Returns:
            <dx1, dx2, ... dxn, dhx, dcx>, where dxi is the gradient tensor for
                the i-th input, its shape is (batch_size,
                input_feature_length). dhx is the gradient for the initial
                hidden state. dcx is the gradient for the initial cell state,
                which is valid only for lstm.
        '''
        if type(flag) is bool:
            if flag:
                flag = model_pb2.kTrain
            else:
                flag = model_pb2.kEval

        tensors = []
        for t in grad:
            assert isinstance(t, tensor.Tensor), 'grad must be py Tensor'
            tensors.append(t.singa_tensor)
        ret = self.layer.BackwardWithMultInputs(flag, tensors)
        return tensor.from_raw_tensors(ret[0]), tensor.from_raw_tensors(ret[1])


class LSTM(RNN):

    def __init__(self, name, hidden_size, dropout=0.0, num_stacks=1,
                 input_mode='linear', bidirectional=False,
                 param_specs=None, input_sample_shape=None):
        super(LSTM, self).__init__(name, hidden_size,  'lstm',  dropout,
                                   num_stacks, input_mode, bidirectional,
                                   param_specs, input_sample_shape)


class GRU(RNN):

    def __init__(self, name, hidden_size, dropout=0.0, num_stacks=1,
                 input_mode='linear', bidirectional=False, param_specs=None,
                 input_sample_shape=None):
        super(GRU, self).__init__(name,  hidden_size, 'gru',  dropout,
                                  num_stacks, input_mode, bidirectional,
                                  param_specs, input_sample_shape)


def _check_engine(engine, allowed_engines):
    assert engine.lower() in Set(allowed_engines), \
        '%s is not a supported engine. Pls use one of %s' % \
        (engine, ', '.join(allowed_engines))


def _create_layer(eng, layer):
    ''' create singa wrap layer.

    Both arguments are case insensitive.
    Args:
        engine, implementation engine, either 'singa' or 'cudnn'
        layer, layer type, e.g., 'convolution', 'pooling'; for activation
        layers, use the specific activation mode, e.g. 'relu', 'tanh'.
    '''
    assert eng != 'cudnn' or cudnn_version > 0, 'CUDNN is not enabled, please '\
        'change the engine, e.g., layer.engine=singacpp'
    layer_type = eng + '_' + layer
    return singa_wrap.CreateLayer(layer_type.lower())


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
            filler.min = specs['low']
            filler.max = specs['high']
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


def _construct_param_specs_from_caffe_proto(lyr_conf):
    """convert the param specs from a caffe layer proto into a singa paramspec
    protobuf object.

    args:
        specs (dict): the fields inlcude
            'name' for parameter name
            'lr_mult' for learning rate multiplier;
            'decay_mult' for weight decay multiplier;
            'init' for init method, which could be 'gaussian', 'uniform',
            'xavier' and 'msra';
            'std', 'mean', 'high', 'low' are used by corresponding init methods;
            caffe model has no 'constraint' and 'regularizer'

    returns:
        a pair of paramspec objects(weight and bias)
    """
    wparam = model_pb2.ParamSpec()
    bparam = model_pb2.ParamSpec()
    if len(lyr_conf.param) > 0:
        wparam.name = lyr_conf.param[0].name
        wparam.lr_mult = lyr_conf.param[0].lr_mult
        wparam.decay_mult = lyr_conf.param[0].decay_mult
        if len(lyr_conf.param) > 1:
            bparam.name = lyr_conf.param[1].name
            bparam.lr_mult = lyr_conf.param[1].lr_mult
            bparam.decay_mult = lyr_conf.param[1].decay_mult
    if wparam.name == '' or wparam.name is None:
        wparam.name = lyr_conf.name + '_weight'
    if bparam.name == '' or bparam.name is None:
        bparam.name = lyr_conf.name + '_bias'
    wfiller = wparam.filler
    bfiller = bparam.filler
    param = ''
    if lyr_conf.type == 'Convolution' or lyr_conf.type == 4:
        param = lyr_conf.convolution_conf
    elif lyr_conf.type == 'InnerProduct' or lyr_conf.type == 14:
        param = lyr_conf.dense_conf

    if param != '':
        wfiller.type = param.weight_filler.type.lower()
        wfiller.min = param.weight_filler.min
        wfiller.max = param.weight_filler.max
        wfiller.mean = param.weight_filler.mean
        wfiller.std = param.weight_filler.std
        wfiller.value = param.weight_filler.value

        bfiller.type = param.bias_filler.type.lower()
        bfiller.min = param.bias_filler.min
        bfiller.max = param.bias_filler.max
        bfiller.mean = param.bias_filler.mean
        bfiller.std = param.bias_filler.std
        bfiller.value = param.bias_filler.value

    return (wparam, bparam)


def get_layer_list():
    """ Return a list of strings which include the identifiers (tags) of all
    supported layers
    """
    return singa_wrap.GetRegisteredLayers()
