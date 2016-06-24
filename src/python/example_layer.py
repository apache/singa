import sys, os

from layer import *

sys.path.append(os.path.join(os.path.dirname(__file__),
                             '..'))
from model_pb2 import *

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


