import sys, os

from .layer import *

from .proto.model_pb2 import *

#---------------------------------------------------------
# example usage
#---------------------------------------------------------

l = Layer('layer')

l_conf = LayerConf()
l_conf.name = "chonho layer"
l.setup(l_conf)

print l.name()

c = Conv2D(2, 3, name='chonho conv')
print c.name()
print c.conf


