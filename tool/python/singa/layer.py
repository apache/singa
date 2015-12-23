#!/usr/bin/env python
from parameter import *
from utils.utility import * 
from utils.message import * 
from google.protobuf import text_format

class Layer(object):
  def __init__(self, **kwargs):
    self.layer = Message('Layer', **kwargs).proto
    # required
    if not 'name' in kwargs:
      setval(self.layer, name=generateName('layer', 1))

    # srclayers are set in Model.build()
    self.is_datalayer = False 

class Data(Layer):
  def __init__(self, load, phase='train', checkpoint=None,
               conf=None, **kwargs):
    assert load != None, 'data type should be specified'
    if load == 'kData':
      super(Data, self).__init__(name=generateName('data'), user_type=load)
    else:
      self.layer_type = enumLayerType(load)
      super(Data, self).__init__(name=generateName('data'), type=self.layer_type)

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
    self.is_datalayer = True

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
    '''
    assert nb_filter > 0 and kernel > 0, 'should be set as positive int'
    super(Convolution2D, self).__init__(name=generateName('conv',1), type=kCConvolution)
    fields = {'num_filters' : nb_filter,
              'kernel' : kernel,
              'stride' : stride,
              'pad' : pad}
    setval(self.layer.convolution_conf, **fields)

    # parameter w  
    if w_param == None:
      self.init = 'gaussian' if init==None else init 
      w_param = Parameter(init=self.init) 
    setParamField(w_param.param, 'w', True, **kwargs)
    setval(self.layer, param=w_param.param)

    # parameter b  
    if b_param == None:
      self.init = 'constant' if init==None else init 
      b_param = Parameter(init=self.init) # default: constant
    setParamField(b_param.param, 'b', True, **kwargs)
    setval(self.layer, param=b_param.param)

    # following layers: e.g., activation, dropout, etc.
    if activation:
      self.mask = Activation(activation=activation).layer

class MaxPooling2D(Layer):
  def __init__(self, pool_size=None, stride=1, ignore_border=True, **kwargs): 
    '''
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
    assert type(pool_size) == tuple and  \
           pool_size[0] == pool_size[1], 'pool size should be square in Singa'
    super(MaxPooling2D, self).__init__(name=generateName('pool'), type=kCPooling, **kwargs)
    fields = {'pool' : PoolingProto().MAX,
              'kernel' : pool_size[0],
              'stride' : stride,
              'pad' : 0 if ignore_border else 1}
    setval(self.layer.pooling_conf, **fields)

class AvgPooling2D(Layer):
  def __init__(self, pool_size=None, stride=1, ignore_border=True, **kwargs): 
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
    assert type(pool_size) == tuple and  \
           pool_size[0] == pool_size[1], 'pool size should be square in Singa'
    super(AvgPooling2D, self).__init__(name=generateName('pool'), type=kCPooling, **kwargs)
    self.layer.pooling_conf.pool = PoolingProto().AVG 
    fields = {'pool' : PoolingProto().AVG,
              'kernel' : pool_size[0],
              'stride' : stride,
              'pad' : 0 if ignore_border else 1}
    setval(self.layer.pooling_conf, **fields)

class LRN2D(Layer):
  def __init__(self, size=0, **kwargs):
    super(LRN2D, self).__init__(name=generateName('norm'), type=kLRN)
    # required
    assert size != 0, 'local size should be set'
    self.layer.lrn_conf.local_size = size 
    init_value = initializations.get('lrn2d', **kwargs)
    setval(self.layer.lrn_conf, **init_value)


class Activation(Layer):
  def __init__(self, activation='stanh', topk=1):
    self.name = activation 
    if activation == 'tanh': activation = 'stanh' # <-- better way to set?
    self.layer_type = enumLayerType(activation)  
    super(Activation, self).__init__(name=generateName(self.name), type=self.layer_type)
    if activation == 'softmaxloss':
      self.layer.softmaxloss_conf.topk = topk

class Dropout(Layer): 
  def __init__(self, ratio=0.5):
    self.name = 'dropout'
    self.layer_type = kDropout
    super(Dropout, self).__init__(name=generateName(self.name), type=self.layer_type)
    self.layer.dropout_conf.dropout_ratio = ratio


class RGB(Layer):
  def __init__(self, meanfile=None, **kwargs):
    assert meanfile != None, 'meanfile should be specified'
    self.name = 'rgb'
    self.layer_type = kRGBImage
    super(RGB, self).__init__(name=generateName(self.name), type=self.layer_type)
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
      init       = (string) // 'unirom', 'gaussian', 'constant'
      **kwargs
        w_lr = (float) // learning rate for w
        w_wd = (float) // weight decay for w
        b_lr = (float) // learning rate for b
        b_wd = (float) // weight decay for b
    '''
    # required
    assert output_dim > 0, 'output_dim should be set'
    super(Dense, self).__init__(type=kInnerProduct, **kwargs)
    self.layer.innerproduct_conf.num_output = output_dim
    if 'transpose' in kwargs:
      self.layer.innerproduct_conf.transpose = kwargs['transpose']
    
    # parameter w (default: gaussian)  
    if w_param == None:
      self.init = 'gaussian' if init==None else init 
      w_param = Parameter(init=self.init) 
    setParamField(w_param.param, 'w', False, **kwargs)
    setval(self.layer, param=w_param.param)

    # parameter b (default: constant) 
    if b_param == None:
      self.init = 'constant' if init==None else init 
      b_param = Parameter(init=self.init)
    setParamField(b_param.param, 'b', False, **kwargs)
    setval(self.layer, param=b_param.param)

    # following layers: e.g., activation, dropout, etc.
    if activation:
      self.mask = Activation(activation=activation).layer


''' Class to deal with multiple layers
'''
class Autoencoder(object):
  def __init__(self, hid_dim=None, out_dim=0, activation=None, 
               param_share=True, **kwargs):
    # required
    assert out_dim >  0, 'out_dim should be set'
    self.out_dim = out_dim
    assert hid_dim != None, 'hid_dim should be set'
    self.hid_dim = [hid_dim] if type(hid_dim)==int else hid_dim 

    self.layer_type = 'AutoEncoder' 
    self.activation = activation
    self.param_share = param_share

class RBM(Layer):
  def __init__(self, out_dim=None, w_param=None, b_param=None, sampling=None, **kwargs):
    '''
    Generate layers (like MLP) according to the number of elements in out_dim, and
      on top of it, two layers RBMVis and RBMHid with bidirectional connection

    required
      out_dim  = (int) or (int list) // the number of hidden nodes
    optional
      sampling = (string)
    '''
    assert out_dim >  0, 'out_dim should be set'
    self.out_dim = [out_dim] if type(out_dim)==int else out_dim 

    self.name = kwargs['name'] if 'name' in kwargs else 'RBMVis' 
    self.layer_type = kwargs['type'] if 'type' in kwargs else kRBMVis
    super(RBM, self).__init__(name=generateName(self.name, withnumber=False), type=self.layer_type)
    setval(self.layer.rbm_conf, hdim=self.out_dim[-1])
    if self.layer_type == kRBMHid and sampling != None: 
      if sampling == 'gaussian':
        setval(self.layer.rbm_conf, gaussian=True)

    # parameter w
    if w_param == None:
      w_param = Parameter(init='gaussian', **kwargs)
      setParamField(w_param.param, 'w', withnumber=False, level=len(self.out_dim), **kwargs)
    else:
      if self.layer_type == kRBMHid:
        del kwargs['name']
      else:
        setParamField(w_param.param, 'w', withnumber=False, level=len(self.out_dim), **kwargs)
    setval(self.layer, param=w_param.param)

    # parameter b
    if b_param == None:
      b_param = Parameter(init='constant', **kwargs)
      setParamField(b_param.param, 'b', withnumber=False, level=len(self.out_dim), **kwargs)
    else:
      if self.layer_type == kRBMHid:
        pass
      else:
        setParamField(b_param.param, 'b', withnumber=False, level=len(self.out_dim), **kwargs)
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
    super(Embedding, self).__init__(name=generateName('embedding',1), user_type='kEmbedding')
    fields = { 'vocab_size': in_dim,
               'word_dim': out_dim }
    setval(self.layer.Extensions[embedding_conf], **fields)
    if w_param == None:
      w_param = Parameter(name=generateName('w'), init=init) # default: uniform
    else:
      setParamField(w_param.param, 'w', True, **kwargs)
    setval(self.layer, param=w_param.param)
    
class RNNLM(Layer):
  def __init__(self, dim, w_param=None, **kwargs):
    super(RNNLM, self).__init__(name=generateName('hidden',1), user_type='kHidden')
    if w_param == None:
      w_param = Parameter(name=generateName('w'), init=init) # default: uniform
    else:
      setParamField(w_param.param, 'w', True, **kwargs)
    setval(self.layer, param=w_param.param)

class UserLossRNNLM(Layer):
  def __init__(self, **kwargs):
    super(UserLossRNNLM, self).__init__(name=generateName('loss',1), user_type='kLoss')
    self.layer.Extensions[loss_conf].nclass = kwargs['nclass'] 
    self.layer.Extensions[loss_conf].vocab_size = kwargs['vocab_size'] 
    setval(self.layer, param=Parameter(name=generateName('w'), init='uniform', scale=0.3).param)
    setval(self.layer, param=Parameter(name=generateName('w',1), init='uniform', scale=0.3).param)


