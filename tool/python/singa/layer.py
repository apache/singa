#!/usr/bin/env python

#/************************************************************
#*
#* Licensed to the Apache Software Foundation (ASF) under one
#* or more contributor license agreements.  See the NOTICE file
#* distributed with this work for additional information
#* regarding copyright ownership.  The ASF licenses this file
#* to you under the Apache License, Version 2.0 (the
#* "License"); you may not use this file except in compliance
#* with the License.  You may obtain a copy of the License at
#*
#*   http://www.apache.org/licenses/LICENSE-2.0
#*
#* Unless required by applicable law or agreed to in writing,
#* software distributed under the License is distributed on an
#* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#* KIND, either express or implied.  See the License for the
#* specific language governing permissions and limitations
#* under the License.
#*
#*************************************************************/

'''
This script includes Layer class and its subclasses that
users can configure different types of layers for their model.
'''

from singa.parameter import Parameter, set_param_field
from singa.initializations import get_init_values
from singa.utils.utility import setval, generate_name
from singa.utils.message import *
from google.protobuf import text_format

class Layer(object):

    def __init__(self, **kwargs):
        '''
        **kwargs (KEY=VALUE)
          partition_dim = (int)  // partition dimension for net
        '''

        self.layer = Message('Layer', **kwargs).proto
        # required
        if not 'name' in kwargs:
            setval(self.layer, name=generate_name('layer', 1))

        # srclayers are set in Model.build()
        self.is_datalayer = False

class Data(Layer):

    def __init__(self, load, phase='train', checkpoint=None,
                 conf=None, **kwargs):
        '''
        required
          load       = (string)  // type of data
        optional
          phase      = (string)  // phase of data layer
          checkpoint = (string)  // checkpoint path
          conf       = (Store)   // Store object
          **kwargs (KEY=VALUE)
            partition_dim = (int)  // partition dimension for net
        '''

        assert load != None, 'data type should be specified'
        if load == 'kData':
            super(Data, self).__init__(name=generate_name('data'),
                                       user_type=load)
        else:
            self.layer_type = enumLayerType(load)
            super(Data, self).__init__(name=generate_name('data'),
                                       type=self.layer_type)
        self.is_datalayer = True

        # include/exclude
        setval(self.layer, include=enumPhase(phase))
        #setval(self.layer, exclude=kTest if phase=='train' else kTrain)

        if conf == None:
            if load == 'kData':
                setval(self.layer.Extensions[data_conf], **kwargs)
            else:
                setval(self.layer.store_conf, **kwargs)
        else:
            setval(self.layer, store_conf=conf.proto)

        self.checkpoint = checkpoint # checkpoint for training data


class Convolution2D(Layer):

    def __init__(self, nb_filter=0, kernel=0, stride=1, pad=0,
                 init=None, w_param=None, b_param=None,
                 activation=None, **kwargs):
        '''
        required
          nb_filter = (int)  // the number of filters
          kernel    = (int)  // the size of filter
        optional
          stride    = (int)  // the size of stride
          pad       = (int)  // the size of padding
          init      = (string)     // 'unirom', 'gaussian', 'constant'
          w_param   = (Parameter)  // Parameter object for weight
          b_param   = (Parameter)  // Parameter object for bias
          **kwargs (KEY=VALUE)
            w_lr = (float) // learning rate multiplier for weight, used to
                           // scale the learning rate when updating parameters.
            w_wd = (float) // weight decay multiplier for weight, used to
                           // scale the weight decay when updating parameters.
            b_lr = (float) // learning rate multiplier for bias 
            b_wd = (float) // weight decay multiplier for bias
        '''

        assert nb_filter > 0 and kernel > 0, 'should be set as positive int'
        super(Convolution2D, self).__init__(name=generate_name('conv', 1),
                                            type=kCConvolution)
        fields = {'num_filters' : nb_filter,
                  'kernel' : kernel,
                  'stride' : stride,
                  'pad' : pad}
        setval(self.layer.convolution_conf, **fields)

        # parameter w
        if w_param == None:
            self.init = 'gaussian' if init == None else init
            w_param = Parameter(init=self.init)
        set_param_field(w_param.param, 'w', True, **kwargs)
        setval(self.layer, param=w_param.param)

        # parameter b
        if b_param == None:
            self.init = 'constant' if init == None else init
            b_param = Parameter(init=self.init) # default: constant
        set_param_field(b_param.param, 'b', True, **kwargs)
        setval(self.layer, param=b_param.param)

        # following layers: e.g., activation, dropout, etc.
        if activation:
            self.mask = Activation(activation=activation).layer

class MaxPooling2D(Layer):

    def __init__(self, pool_size=None,
                 stride=1, ignore_border=True, **kwargs):
        '''
        Max Pooling layer

        required
          pool_size     = (int|tuple) // the size for pooling
        optional
          stride        = (int)       // the size of striding
          ignore_border = (bool)      // flag for padding
          **kwargs                    // fields for Layer class
        '''

        assert pool_size != None, 'pool_size is required'
        if type(pool_size) == int:
            pool_size = (pool_size, pool_size)
        assert type(pool_size) == tuple and pool_size[0] == pool_size[1], \
               'pool size should be square in Singa'
        super(MaxPooling2D, self).__init__(name=generate_name('pool'),
                                           type=kCPooling, **kwargs)
        fields = {'pool' : PoolingProto().MAX,
                  'kernel' : pool_size[0],
                  'stride' : stride,
                  'pad' : 0 if ignore_border else 1}
        setval(self.layer.pooling_conf, **fields)

class AvgPooling2D(Layer):

    def __init__(self, pool_size=None,
                 stride=1, ignore_border=True, **kwargs):
        '''
        required
          pool_size     = (int|tuple) // size for pooling
        optional
          stride        = (int)       // size of striding
          ignore_border = (bool)      // flag for padding
          **kwargs                    // fields for Layer class
        '''

        assert pool_size != None, 'pool_size is required'
        if type(pool_size) == int:
            pool_size = (pool_size, pool_size)
        assert type(pool_size) == tuple and pool_size[0] == pool_size[1], \
               'pool size should be square in Singa'
        super(AvgPooling2D, self).__init__(name=generate_name('pool'),
                                           type=kCPooling, **kwargs)
        self.layer.pooling_conf.pool = PoolingProto().AVG
        fields = {'pool' : PoolingProto().AVG,
                  'kernel' : pool_size[0],
                  'stride' : stride,
                  'pad' : 0 if ignore_border else 1}
        setval(self.layer.pooling_conf, **fields)

class LRN2D(Layer):

    def __init__(self, size=0, **kwargs):
        '''
        required
          size = (int)  // local size
        '''

        super(LRN2D, self).__init__(name=generate_name('norm'), type=kLRN)
        # required
        assert size != 0, 'local size should be set'
        self.layer.lrn_conf.local_size = size
        init_values = get_init_values('lrn2d', **kwargs)
        setval(self.layer.lrn_conf, **init_values)


class Activation(Layer):

    def __init__(self, activation='stanh', topk=1):
        '''
        required
          activation = (string)
        optional
          topk       = (int)  // the number of results
        '''

        self.name = activation
        if activation == 'tanh': activation = 'stanh' # <-- better way to set?
        self.layer_type = enumLayerType(activation)
        super(Activation, self).__init__(name=generate_name(self.name),
                                         type=self.layer_type)
        if activation == 'softmaxloss':
            self.layer.softmaxloss_conf.topk = topk

class Dropout(Layer):

    def __init__(self, ratio=0.5):
        '''
        required
          ratio = (float) // ratio of drop out nodes
        '''

        self.name = 'dropout'
        self.layer_type = enumLayerType(self.name)
        super(Dropout, self).__init__(name=generate_name(self.name),
                                      type=self.layer_type)
        self.layer.dropout_conf.dropout_ratio = ratio


class RGB(Layer):

    def __init__(self, meanfile=None, **kwargs):
        '''
        required
          meanfile = (string) // path to meanfile (depreciated)
        '''

        assert meanfile != None, 'meanfile should be specified'
        self.name = 'rgb'
        self.layer_type = kRGBImage
        super(RGB, self).__init__(name=generate_name(self.name),
                                  type=self.layer_type)
        self.layer.rgbimage_conf.meanfile = meanfile

class Dense(Layer):

    def __init__(self, output_dim=0, activation=None,
                 init=None, w_param=None, b_param=None, input_dim=None,
                 **kwargs):
        '''
        required
          output_dim = (int)
        optional
          activation = (string)
          init       = (string)     // 'unirom', 'gaussian', 'constant'
          w_param    = (Parameter)  // Parameter object for weight
          b_param    = (Parameter)  // Parameter object for bias
          **kwargs
            w_lr = (float) // learning rate multiplier for weight, used to
                           // scale the learning rate when updating parameters.
            w_wd = (float) // weight decay multiplier for weight, used to
                           // scale the weight decay when updating parameters.
            b_lr = (float) // learning rate multiplier for bias 
            b_wd = (float) // weight decay multiplier for bias
        '''
        # required
        assert output_dim > 0, 'output_dim should be set'
        super(Dense, self).__init__(type=kInnerProduct, **kwargs)
        self.layer.innerproduct_conf.num_output = output_dim
        if 'transpose' in kwargs:
            self.layer.innerproduct_conf.transpose = kwargs['transpose']

        # parameter w (default: gaussian)
        if w_param == None:
            self.init = 'gaussian' if init == None else init
            w_param = Parameter(init=self.init)
        set_param_field(w_param.param, 'w', False, **kwargs)
        setval(self.layer, param=w_param.param)

        # parameter b (default: constant)
        if b_param == None:
            self.init = 'constant' if init == None else init
            b_param = Parameter(init=self.init)
        set_param_field(b_param.param, 'b', False, **kwargs)
        setval(self.layer, param=b_param.param)

        # following layers: e.g., activation, dropout, etc.
        if activation:
            self.mask = Activation(activation=activation).layer


''' Classes to deal with multiple layers
'''
class Autoencoder(object):

    def __init__(self, hid_dim=None, out_dim=0,
                 activation=None, param_share=True):
        '''
        Generate a set of layers (like MLP) for encoder and decoder
        The layers are expanded and added in Sequential.add()

        required
          hid_dim     = (int/list) // the number of nodes in hidden layers
          out_dim     = (int)      // the number of nodes in the top layer
        optional 
          activation  = (string)
          param_share = (bool)     // to share params in encoder and decoder
        '''

        # required
        assert out_dim > 0, 'out_dim should be set'
        self.out_dim = out_dim
        assert hid_dim != None, 'hid_dim should be set'
        self.hid_dim = [hid_dim] if type(hid_dim) == int else hid_dim

        self.layer_type = 'AutoEncoder'
        self.activation = activation
        self.param_share = param_share

class RBM(Layer):

    def __init__(self, out_dim=None, w_param=None, b_param=None,
                 sampling=None, **kwargs):
        '''
        Generate a set of layers (like MLP) according to the number of elements
          in out_dim, and on top of it, two layers RBMVis and RBMHid with
          bidirectional connection
        The layers are expanded and added in Energy.add()

        required
          out_dim  = (int) or (int list) // the number of hidden nodes
        optional
          w_param  = (Parameter)  // Parameter object for weight
          b_param  = (Parameter)  // Parameter object for bias
          sampling = (string)
        '''

        assert out_dim > 0, 'out_dim should be set'
        self.out_dim = [out_dim] if type(out_dim) == int else out_dim

        self.name = kwargs['name'] if 'name' in kwargs else 'RBMVis'
        self.layer_type = kwargs['type'] if 'type' in kwargs else kRBMVis
        super(RBM, self).__init__(name=generate_name(self.name,
                                  withnumber=False), type=self.layer_type)
        setval(self.layer.rbm_conf, hdim=self.out_dim[-1])
        if self.layer_type == kRBMHid and sampling != None:
            if sampling == 'gaussian':
                setval(self.layer.rbm_conf, gaussian=True)

        # parameter w
        if w_param == None:
            w_param = Parameter(init='gaussian', **kwargs)
            set_param_field(w_param.param, 'w', withnumber=False,
                            level=len(self.out_dim), **kwargs)
        else:
            if self.layer_type == kRBMHid:
                del kwargs['name']
            else:
                set_param_field(w_param.param, 'w', withnumber=False,
        	  	        level=len(self.out_dim), **kwargs)
        setval(self.layer, param=w_param.param)

        # parameter b
        if b_param == None:
            b_param = Parameter(init='constant', **kwargs)
            set_param_field(b_param.param, 'b', withnumber=False,
        		    level=len(self.out_dim), **kwargs)
        else:
            if self.layer_type == kRBMHid:
                pass
            else:
                set_param_field(b_param.param, 'b', withnumber=False,
        		        level=len(self.out_dim), **kwargs)
        setval(self.layer, param=b_param.param)

        if self.layer_type == kRBMVis:
            wname = w_param.param.name
            parw = Parameter(name=wname+"_", init='none', share_from=wname)
            bname = b_param.param.name
            parb = Parameter(name=bname+"2", wd=0, init='constant')
            self.bidirect = RBM(self.out_dim, name='RBMHid', type=kRBMHid,
                         w_param=parw, b_param=parb, sampling=sampling).layer

class Embedding(Layer):

    def __init__(self, in_dim, out_dim, w_param=None, **kwargs):

        super(Embedding, self).__init__(name=generate_name('embedding', 1),
                                        user_type='kEmbedding')
        fields = {'vocab_size': in_dim,
                  'word_dim': out_dim}
        setval(self.layer.Extensions[embedding_conf], **fields)
        if w_param == None:
            # default: uniform
            w_param = Parameter(name=generate_name('w'), init=init)
        else:
            set_param_field(w_param.param, 'w', True, **kwargs)
        setval(self.layer, param=w_param.param)

class RNNLM(Layer):

    def __init__(self, dim, w_param=None, **kwargs):

        super(RNNLM, self).__init__(name=generate_name('hidden', 1),
                                    user_type='kHidden')
        if w_param == None:
            # default: uniform
            w_param = Parameter(name=generate_name('w'), init=init)
        else:
            set_param_field(w_param.param, 'w', True, **kwargs)
        setval(self.layer, param=w_param.param)

class UserLossRNNLM(Layer):

    def __init__(self, **kwargs):

        super(UserLossRNNLM, self).__init__(name=generate_name('loss', 1),
                                            user_type='kLoss')
        self.layer.Extensions[loss_conf].nclass = kwargs['nclass']
        self.layer.Extensions[loss_conf].vocab_size = kwargs['vocab_size']
        setval(self.layer, param=Parameter(name=generate_name('w'),
                                           init='uniform', scale=0.3).param)
        setval(self.layer, param=Parameter(name=generate_name('w', 1),
                                           init='uniform', scale=0.3).param)
