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
import numpy
from singa.parameter import Parameter, set_param_field
from singa.initializations import get_init_values
from singa.utils.utility import setval, generate_name
from singa.utils.message import *
from google.protobuf import text_format

from singa.driver import Layer as SingaLayer, Updater as SingaUpdater,\
                         intVector, floatVector, layerVector,\
                         paramVector, floatArray_frompointer, DummyLayer

class Layer(object):

    singaupdater = None

    def __init__(self, **kwargs):
        '''
        **kwargs (KEY=VALUE)
          partition_dim = (int)  // partition dimension for net
        '''

        self.layer = Message('Layer', **kwargs).proto
        # required field
        if not 'name' in kwargs:
            setval(self.layer, name=generate_name('layer', 1))

        # layer connectivity is set in Model.build()
        self.is_datalayer = False
        self.singalayer = None
        self.srclayers = [] 


        # set src for Rafiki
        if 'src' in kwargs:
            self.src = kwargs['src']
        else:
            self.src = None

    def setup(self, srclys):
        # create singa::Layer and store srclayers 
        if self.singalayer == None:
            self.singalayer = SingaLayer.CreateLayer(self.layer.SerializeToString())  
            self.singaSrclayerVector = layerVector(len(srclys)) 
            for i in range(len(srclys)):
                self.srclayers.append(srclys[i])
                self.singaSrclayerVector[i] = srclys[i].get_singalayer()
            # set up the layer
            SingaLayer.SetupLayer(self.singalayer, self.layer.SerializeToString(), self.singaSrclayerVector)

    def ComputeFeature(self, *srclys):
        ''' The method creates and sets up singa::Layer
            and maintains its source layers
            then call ComputeFeature for data transformation.

            *srclys = (list)  // a list of source layers
        '''

        # create singa::Layer and store srclayers 
        if self.singalayer == None:
            if self.src != None:
                srclys = self.src
            self.singalayer = SingaLayer.CreateLayer(self.layer.SerializeToString())  
            self.singaSrclayerVector = layerVector(len(srclys)) 
            for i in range(len(srclys)):
                self.srclayers.append(srclys[i])
                self.singaSrclayerVector[i] = srclys[i].get_singalayer()
            # set up the layer
            SingaLayer.SetupLayer(self.singalayer, self.layer.SerializeToString(), self.singaSrclayerVector)

        self.singalayer.ComputeFeature(1, self.singaSrclayerVector)

    def ComputeGradient(self, step, upd=None):
        ''' The method creates singa::Updater
            and calls ComputeGradient for gradient computation
            then updates the parameters.

            step = (int)    // a training step
            upd = (object)  // Updater object
        '''

        # create singa::Updater
        assert upd != None, 'required Updater (see model.py)' 
        if Layer.singaupdater == None:
            Layer.singaupdater = SingaUpdater.CreateUpdater(upd.proto.SerializeToString()) 

        # call ComputeGradient of Singa
        self.singalayer.ComputeGradient(1, self.singaSrclayerVector)

        # update parameters
        singaParams = self.singalayer.GetParams()
        for p in singaParams:
            Layer.singaupdater.Update(step, p, 1.0)

        # recursively call ComputeGradient of srclayers
        #(TODO) what if there are multiple source layers???
        for sly in self.srclayers:
            if sly.srclayers != None:
                sly.ComputeGradient(step, upd) 

    def GetParams(self):
        ''' The method gets parameter values
            singaParams[0] for weight
            singaParams[1] for bias
        '''
        singaParams = self.singalayer.GetParams()
        assert len(singaParams) == 2, 'weight and bias'
        # for weight
        weight_array = floatArray_frompointer(singaParams[0].mutable_cpu_data())
        weight = [ weight_array[i] for i in range(singaParams[0].size()) ]
        weight = numpy.array(weight).reshape(singaParams[0].shape())
        # for bias
        bias_array = floatArray_frompointer(singaParams[1].mutable_cpu_data())
        bias = [ bias_array[i] for i in range(singaParams[1].size()) ]
        bias = numpy.array(bias).reshape(singaParams[1].shape()[0], 1)

        return weight, bias

    def SetParams(self, *params):
        ''' The method sets parameter values
            params[0] for weight
            params[1] for bias
        '''
        singaParams = self.singalayer.GetParams()
        import pb2.common_pb2 as cm
        for k in range(len(params)):
            bp = cm.BlobProto()
            bp.shape.append(int(params[k].shape[0]))
            bp.shape.append(int(params[k].shape[1]))
            for i in range(params[k].shape[0]):
                for j in range(params[k].shape[1]):
                    bp.data.append(params[k][i,j])
            singaParams[k].FromProto(bp.SerializeToString())

    def GetData(self):
        blobptr = self.singalayer.data(self.singalayer)
        data_array = floatArray_frompointer(blobptr.mutable_cpu_data())
        data = [ data_array[i] for i in range(blobptr.count()) ]
        return data

    def display(self):
        debug, flag = 0, 0
        print self.singalayer.ToString(debug, flag)

    def get_singalayer(self):
        return self.singalayer


class Dummy(object):

    def __init__(self, shape=[], path='', dtype='', src=[]):
        ''' Dummy layer is used for data layer
            shape = (list)   // [# of samples, # of channels, img h, img w]
            path  = (string) // path to dataset
        '''
        self.is_datalayer = True
        self.srclayers = None 
        self.singalayer = None

        # create layer proto for Dummy layer
        kwargs = {'name':'dummy', 'type':kDummy}
        self.layer = Message('Layer', **kwargs).proto


        # if dataset path is not specified, skip
        # otherwise, load dataset
        if path == '':
            return

        self.shape = shape
        self.path = path
        self.src = None
        self.batch_index = 0

        import numpy as np
        nb_samples = shape[0]
        nb_pixels = shape[1]
        for i in range(len(shape)-2):
            nb_pixels *= shape[i+2]  
        if dtype=='byte': 
            self.is_label = 0
            d = np.fromfile(path, dtype=np.uint8)
        elif dtype=='int': 
            self.is_label = 1
            d = np.fromfile(path, dtype=np.int)
        self.data = d.reshape(nb_samples, nb_pixels)


    def setup(self, data_shape):
        ''' Create and Setup singa Dummy layer
            called by load_model_parameter
        '''
        if self.singalayer == None:
            setval(self.layer.dummy_conf, input=True)
            setval(self.layer.dummy_conf, shape=data_shape)
            self.singalayer = DummyLayer()
            self.singalayer.Setup(self.layer.SerializeToString(), layerVector(0))


    def FetchData(self, batchsize):

        d = self.data[self.batch_index*batchsize:(self.batch_index+1)*batchsize, :]
        self.Feed(d, self.shape[1], self.is_label)
        self.batch_index += 1


    def Feed(self, data, nb_channel=1, is_label=0):
        ''' Create and Setup singa::DummyLayer for input data
            Insert data using Feed()
        '''

        batchsize, hdim = data.shape
        datasize = batchsize * hdim
        imgsize = int(numpy.sqrt(hdim/nb_channel)) 
        shapeVector = [batchsize, nb_channel, imgsize, imgsize] 

        # create and setup the dummy layer
        if self.singalayer == None:
            setval(self.layer.dummy_conf, input=True)
            setval(self.layer.dummy_conf, shape=shapeVector)
            self.singalayer = DummyLayer()
            self.singalayer.Setup(self.layer.SerializeToString(), layerVector(0))

        # feed input data
        data = data.astype(numpy.float) 
        dataVector = floatVector(datasize)
        k = 0
        for i in range(batchsize):
            for j in range(hdim):
                dataVector[k] = data[i,j]
                k += 1
        self.singalayer.Feed(shapeVector, dataVector, is_label)

    def get_singalayer(self):
        return self.singalayer.ToLayer()

class ImageData(object):

    def __init__(self, shape=[], data_path='', data_type='byte',mean_path='',mean_type='float'):
        ''' Dummy layer is used for data layer
            shape = (list)   // [# of samples, # of channels, img h, img w]
            data_path  = (string) // path to dataset
            mean_path
        '''
        self.is_datalayer = True
        self.srclayers = None 
        self.singalayer = None
        self.is_label = False 
        # create layer proto for Dummy layer
        kwargs = {'name':'dummy', 'type':kDummy}
        self.layer = Message('Layer', **kwargs).proto

        # if dataset path is not specified, skip
        # otherwise, load dataset
        if data_path == '' or mean_path=='':
            return

        self.shape = shape
        self.data_path = data_path
        self.mean_path = mean_path
        self.src = None
        self.batch_index = 0

        import numpy as np
        nb_samples = shape[0]
        nb_pixels = shape[1]
        for i in range(len(shape)-2):
            nb_pixels *= shape[i+2]  

        if data_type=='byte': 
            d = np.fromfile(data_path, dtype=np.uint8)
        elif data_type=='int': 
            d = np.fromfile(data_path, dtype=np.int)
        self.data = d.reshape(nb_samples, nb_pixels)

        if mean_type=='float': 
            d = np.fromfile(mean_path, dtype=np.float32)
        self.mean = d.reshape(1, nb_pixels)

    def setup(self, data_shape):
        ''' Create and Setup singa Dummy layer
            called by load_model_parameter
        '''
        if self.singalayer == None:
            setval(self.layer.dummy_conf, input=True)
            setval(self.layer.dummy_conf, shape=data_shape)
            self.singalayer = DummyLayer()
            self.singalayer.Setup(self.layer.SerializeToString(), layerVector(0))


    def FetchData(self, batchsize):

        d = self.data[self.batch_index*batchsize:(self.batch_index+1)*batchsize, :]
        self.Feed(d, self.shape[1])
        self.batch_index += 1
        if (self.batch_index+1)*batchsize>self.data.shape[0]:
            self.batch_index=0



    def Feed(self, data, nb_channel=1):
        ''' Create and Setup singa::DummyLayer for input data
            Insert data using Feed()
            Need to minus the mean file
        '''
        batchsize, hdim = data.shape
        datasize = batchsize * hdim
        imgsize = int(numpy.sqrt(hdim/nb_channel)) 
        shapeVector = [batchsize, nb_channel, imgsize, imgsize] 
        #print shapeVector
        # create and setup the dummy layer
        if self.singalayer == None:
            setval(self.layer.dummy_conf, input=True)
            setval(self.layer.dummy_conf, shape=shapeVector)
            self.singalayer = DummyLayer()
            self.singalayer.Setup(self.layer.SerializeToString(), layerVector(0))

        # feed input data and minus mean 
        data = data.astype(numpy.float) 
        dataVector = floatVector(datasize)
        k = 0
        for i in range(batchsize):
            for j in range(hdim):
                dataVector[k] = data[i,j]-self.mean[0,j]
                k += 1
        self.singalayer.Feed(shapeVector, dataVector, 0)

    def get_singalayer(self):
        return self.singalayer.ToLayer()


class LabelData(object):

    def __init__(self, shape=[], label_path='', label_type='int'):
        ''' Dummy layer is used for label data layer
            shape = (list)   // [# of samples, # of channels, img h, img w]
            data_path  = (string) // path to dataset
            mean_path
        '''
        self.is_datalayer = True
        self.srclayers = None 
        self.singalayer = None
        self.is_label = True
        # create layer proto for Dummy layer
        kwargs = {'name':'dummy', 'type':kDummy}
        self.layer = Message('Layer', **kwargs).proto

        # if dataset path is not specified, skip
        # otherwise, load dataset
        if label_path == '':
            return

        self.shape = shape
        self.label_path = label_path
        self.src = None
        self.batch_index = 0

        import numpy as np
        nb_samples = shape[0]

        if label_type=='int': 
            d = np.fromfile(label_path, dtype=np.int)
        self.data = d.reshape(nb_samples, 1)

    def setup(self, data_shape):
        ''' Create and Setup singa Dummy layer
            called by load_model_parameter
        '''
        if self.singalayer == None:
            setval(self.layer.dummy_conf, input=True)
            setval(self.layer.dummy_conf, shape=data_shape)
            self.singalayer = DummyLayer()
            self.singalayer.Setup(self.layer.SerializeToString(), layerVector(0))


    def FetchData(self, batchsize):

        d = self.data[self.batch_index*batchsize:(self.batch_index+1)*batchsize, :]
        self.Feed(d, self.shape[1])
        self.batch_index += 1
        if (self.batch_index+1)*batchsize>self.data.shape[0]:
            self.batch_index=0

    def Feed(self, data,nb_chanel=1):
        ''' Create and Setup singa::DummyLayer for input data
            Insert data using Feed()
            Need to minus the mean file
        '''
        batchsize = data.shape[0]
        shapeVector = [batchsize, 1] 

        # create and setup the dummy layer
        if self.singalayer == None:
            setval(self.layer.dummy_conf, input=True)
            setval(self.layer.dummy_conf, shape=shapeVector)
            self.singalayer = DummyLayer()
            self.singalayer.Setup(self.layer.SerializeToString(), layerVector(0))

        data = data.astype(numpy.float) 
        dataVector = floatVector(batchsize)
        for i in range(batchsize):
            dataVector[i] = data[i,0]
        self.singalayer.Feed(shapeVector, dataVector, 1)

    def get_singalayer(self):
        return self.singalayer.ToLayer()

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
                                       user_type=load, **kwargs)
        else:
            self.layer_type = enumLayerType(load)
            super(Data, self).__init__(name=generate_name('data'),
                                       type=self.layer_type, **kwargs)
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
          nb_filter = (int)        // the number of filters
          kernel    = (int/tuple)  // the size of filter
        optional
          stride    = (int/tuple)  // the size of stride
          pad       = (int/tuple)  // the size of padding
          init      = (string)     // 'uniform', 'gaussian', 'constant'
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

        assert nb_filter > 0, 'nb_filter should be set as positive int'
        super(Convolution2D, self).__init__(name=generate_name('conv', 1),
                                            type=kCConvolution, **kwargs)
        fields = {"num_filters":nb_filter}
        # for kernel
        if type(kernel) == int:
          fields['kernel'] = kernel
        else:
          fields['kernel_x'] = kernel[0]
          fields['kernel_y'] = kernel[1]
        # for stride 
        if type(stride) == int:
          fields['stride'] = stride
        else:
          fields['stride_x'] = stride[0]
          fields['stride_y'] = stride[1]
        # for pad 
        if type(pad) == int:
          fields['pad'] = pad 
        else:
          fields['pad_x'] = pad[0]
          fields['pad_y'] = pad[1]

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
               'currently pool size should be square in Singa'
        super(MaxPooling2D, self).__init__(name=generate_name('pool'),
                                           type=kCPooling, **kwargs)
        fields = {'pool' : MAX,
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
               'currently pool size should be square in Singa'
        super(AvgPooling2D, self).__init__(name=generate_name('pool'),
                                           type=kCPooling, **kwargs)
        self.layer.pooling_conf.pool = AVG
        fields = {'pool' : AVG,
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

        super(LRN2D, self).__init__(name=generate_name('norm'), type=kLRN, **kwargs)
        # required
        assert size != 0, 'local size should be set'
        self.layer.lrn_conf.local_size = size
        init_values = get_init_values('lrn2d', **kwargs)
        setval(self.layer.lrn_conf, **init_values)

class Loss(Layer):

    def __init__(self, lossname, topk=1, **kwargs):
        '''
        required
          lossname = (string) // softmaxloss, euclideanloss
        '''
        self.layer_type = enumLayerType(lossname)
        super(Loss, self).__init__(name=generate_name(lossname),
                                         type=self.layer_type, **kwargs)
        if lossname == 'softmaxloss':
            self.layer.softmaxloss_conf.topk = topk

class Activation(Layer):

    def __init__(self, activation='stanh', **kwargs):
        '''
        required
          activation = (string) // relu, sigmoid, tanh, stanh, softmax.
        '''
        if activation == 'tanh':
          print 'Warning: Tanh layer is not supported for CPU'

        self.name = activation
        self.layer_type = kActivation
        if activation == 'stanh':
            self.layer_type = kSTanh
        elif activation == 'softmax':
            self.layer_type = kSoftmax
        super(Activation, self).__init__(name=generate_name(self.name),
                                         type=self.layer_type, **kwargs)
        if activation == 'relu':
            self.layer.activation_conf.type = RELU
        elif activation == 'sigmoid':
            self.layer.activation_conf.type = SIGMOID
        elif activation == 'tanh':
            self.layer.activation_conf.type = TANH # for GPU
        #elif activation == 'stanh':
        #    self.layer.activation_conf.type = STANH
        


class Dropout(Layer):

    def __init__(self, ratio=0.5):
        '''
        required
          ratio = (float) // ratio of drop out nodes
        '''

        self.name = 'dropout'
        self.layer_type = enumLayerType(self.name)
        super(Dropout, self).__init__(name=generate_name(self.name),
                                      type=self.layer_type, **kwargs)
        self.layer.dropout_conf.dropout_ratio = ratio

class Accuracy(Layer):

    def __init__(self, **kwargs):
        '''
        '''

        self.name = 'accuracy'
        self.layer_type = enumLayerType(self.name)
        super(Accuracy, self).__init__(name=generate_name(self.name),
                                       type=self.layer_type, **kwargs)

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
          init       = (string)     // 'uniform', 'gaussian', 'constant'
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
                                  withnumber=False), type=self.layer_type, **kwargs)
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
