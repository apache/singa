#!/usr/bin/env python 
import sys, os 
from utility import * 
sys.path.append(os.path.join(os.path.dirname(__file__),'../../pb2')) 

module_list=[]

# import all modules in dir singa_root/too/pb2, except common, singa and __init__
for f in os.listdir(os.path.join(os.path.dirname(__file__),'../../pb2')):
  if (f.endswith(".pyc")):
    continue
  if(f == "__init__.py" or f == "common_pb2.py" or f == "singa_pb2.py" ):
    continue
  module_name = f.split('.')[0]
  module=__import__(module_name)  
  module_list.append(module)
  for func_name in dir(module):
    if not func_name.startswith("__"):
      globals()[func_name] = getattr(module,func_name)

class Message(object):
  def __init__(self,protoname,**kwargs):
    for module in module_list:
      if hasattr(module,protoname+"Proto"):
        class_ = getattr(module,protoname+"Proto")
        self.proto = class_()
        return setval(self.proto,**kwargs)
    raise Exception('invalid protoname')

enumDict_=dict()

#get all enum type list in modules
for module in module_list:
  for enumtype in module.DESCRIPTOR.enum_types_by_name:
    tempDict=enumDict_[enumtype]=dict()
    for name in getattr(module,enumtype).DESCRIPTOR.values_by_name: 
      tempDict[name[1:].lower()]=getattr(module,name)

def make_function(enumtype):
  def _function(key):
    return enumDict_[enumtype][key]
  return _function

current_module = sys.modules[__name__]

#def all the enumtypes
for module in module_list:
  for enumtype in module.DESCRIPTOR.enum_types_by_name:
    setattr(current_module,"enum"+enumtype,make_function(enumtype))

