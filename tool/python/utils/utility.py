#!/usr/bin/env python

layerid = 0 
paramid = 0 

def generateName(label, op=0):
  global layerid, paramid
  num = layerid
  if label == 'layer':
    if op ==1: layerid += 1
    num = layerid
  elif label == 'param':
    if op ==1: paramid += 1
    num = paramid
  else:
    if op ==1: layerid += 1
    num = layerid

  return '{0}{1}'.format(label, num)


def setval(proto, **kwargs):
  for k,v in kwargs.items():
    #print 'kv: ', k, ', ', v
    if hasattr(proto, k):
      flabel = proto.DESCRIPTOR.fields_by_name[k].label
      ftype  = proto.DESCRIPTOR.fields_by_name[k].type

      fattr  = getattr(proto, k) 
      if flabel == 3: # repeated field
        if ftype == 11: # message type 
          fattr = fattr.add()
          fattr.MergeFrom(v)
        else:
          if type(v) == list or type(v) == tuple:
            for i in range(len(v)):
              fattr.append(v[i])
          else:
            fattr.append(v)
      else:
        if ftype == 11: # message type 
          fattr = getattr(proto,k)
          fattr.MergeFrom(v)
        else:
          setattr(proto, k, v)
