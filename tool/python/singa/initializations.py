#!/usr/bin/env python

def get(identifier, **kwargs):

  field = {}

  if identifier == 'none':
    return
  
  if identifier == 'uniform':
    scale = kwargs['scale'] if 'scale' in kwargs else 0.05 
    names = ['low', 'high']
    values = [-scale, scale]

  elif identifier == 'constant':
    names = ['value']
    values = [0]

  elif identifier == 'gaussian':
    names = ['mean', 'std']
    values = [0, 0.01]

  elif identifier == 'conv2d':
    names = ['stride', 'pad']
    values = [1, 0]

  elif identifier == 'lrn2d':
    names = ['alpha', 'beta', 'knorm']
    values = [1, 0.75, 1]

  for i in range(len(names)):
    field[names[i]] = kwargs[names[i]] if names[i] in kwargs else values[i]
 
  return field
