#!/usr/bin/env python
import initializations
from utils.utility import * 
from utils.message import * 
from google.protobuf import text_format

class Parameter(object):

  def __init__(self, **kwargs):
    '''
    optional
      **kwargs
        name  = (string) // parameter name
        lr    = (float)  // learning rate
        wd    = (float)  // weight decay
        init  = (string) // initialization type {'constant','uniform','gaussian'} 
        value = (int)    // value for 'constant'
        scale = (float)  // [low, high] for 'uniform', low=-scale, high=scale
        low   = (float)  // low value   for 'uniform'
        high  = (float)  // high value  for 'uniform' 
        mean  = (float)  // mean for 'gaussian'
        std   = (float)  // std  for 'gaussian'
    '''
    fields = {'lr_scale' : kwargs['lr'] if 'lr' in kwargs else 1,
              'wd_scale' : kwargs['wd'] if 'wd' in kwargs else 1
             }
    self.param = Message('Param', **fields).proto

    if not 'name' in kwargs:
      setval(self.param, name=generateName('param', 1))
    else:
      pname = kwargs['name']
      # parameter name for RBM
      if 'level' in kwargs:
        pname += str(kwargs['level'])
        if pname[0] == 'b':
          pname += '2'
      setval(self.param, name=pname)

    if 'share_from' in kwargs:
      setval(self.param, share_from=kwargs['share_from'])

    if 'init' in kwargs:
      init_values = initializations.get(kwargs['init'], **kwargs)

      if not kwargs['init'] == 'none':
        pg = Message('ParamGen', type=enumInitMethod(kwargs['init']), **init_values)
        del kwargs['init']
        setval(self.param, init=pg.proto)
    else: # default: uniform
      pg = Message('ParamGen', type=enumInitMethod('uniform'))
      setval(self.param, init=pg.proto)

  def update(self, **fields):
    setval(self.param, **fields) 
    setval(self.param.init, **fields) 


def setParamField(param, pname, changename=False, withnumber=True, **kwargs):
  ''' param      = (ParamProto)
      pname      = (string)     // 'w' for wiehgt, or 'b' for bias
      changename = (bool)       // update parameter name if True
      withnumber = (bool)       // add layer number if True
      **kwargs
  '''
  assert pname == 'w' or pname == 'b', 'pname should be w or b'

  lr = param.lr_scale
  wd = param.wd_scale
  initkv = {}

  if pname == 'w':
    if 'w_lr' in kwargs:
      lr = kwargs['w_lr'] 
      del kwargs['w_lr']
    if 'w_wd' in kwargs:
      wd = kwargs['w_wd']
      del kwargs['w_wd']
    for k, v in kwargs.items():
      if k.startswith('w_'): 
        initkv[k[2:]] = v 

  elif pname == 'b':
    if 'b_lr' in kwargs:
      lr = kwargs['b_lr']
      del kwargs['b_lr']
    if 'b_wd' in kwargs:
      wd = kwargs['b_wd']
      del kwargs['b_wd']
    for k, v in kwargs.items():
      if k.startswith('b_'): 
        initkv[k[2:]] = v 

  field = {'lr_scale' : lr, 'wd_scale' : wd}

  # Set/update parameter fields
  if param.name.startswith('param') or changename==True:
    if 'level' in kwargs:  # parameter name for RBM
      pname += str(kwargs['level'])
    setval(param, name=generateName(pname, withnumber=withnumber), **field)
  else:
    setval(param, **field)

  # Set/update parameter init fields
  setval(param.init, **initkv)
