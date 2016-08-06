import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))
from singa import layer
from singa import metric
from singa import loss
from singa import net as ffnet
from singa.proto import core_pb2

def ConvBnReLU(net, name, nb_filers, sample_shape=None):
    net.add(layer.Conv2D(name + '_1', nb_filers, 3, 1, pad=1,
                         input_sample_shape=sample_shape))
    net.add(layer.BatchNormalization(name + '_2'))
    net.add(layer.Activation(name + '_3'))

def create_net():
    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
    ConvBnReLU(net, 'conv1_1', 64, (3, 32, 32))
    net.add(layer.Dropout('drop1', 0.3, engine='cudnn'))
    ConvBnReLU(net, 'conv1_2', 64)
    net.add(layer.MaxPooling2D('pool1', 2, 2, border_mode='valid'))
    ConvBnReLU(net, 'conv2_1', 128)
    net.add(layer.Dropout('drop2_1', 0.4, engine='cudnn'))
    ConvBnReLU(net, 'conv2_2', 128)
    net.add(layer.MaxPooling2D('pool2', 2, 2, border_mode='valid'))
    ConvBnReLU(net, 'conv3_1', 256)
    net.add(layer.Dropout('drop3_1', 0.4, engine='cudnn'))
    ConvBnReLU(net, 'conv3_2', 256)
    net.add(layer.Dropout('drop3_2', 0.4, engine='cudnn'))
    ConvBnReLU(net, 'conv3_3', 256)
    net.add(layer.MaxPooling2D('pool3', 2, 2, border_mode='valid'))
    ConvBnReLU(net, 'conv4_1', 512)
    net.add(layer.Dropout('drop4_1', 0.4, engine='cudnn'))
    ConvBnReLU(net, 'conv4_2', 512)
    net.add(layer.Dropout('drop4_2', 0.4, engine='cudnn'))
    ConvBnReLU(net, 'conv4_3', 512)
    net.add(layer.MaxPooling2D('pool4', 2, 2, border_mode='valid'))
    ConvBnReLU(net, 'conv5_1', 512)
    net.add(layer.Dropout('drop5_1', 0.4, engine='cudnn'))
    ConvBnReLU(net, 'conv5_2', 512)
    net.add(layer.Dropout('drop5_2', 0.4, engine='cudnn'))
    ConvBnReLU(net, 'conv5_3', 512)
    net.add(layer.MaxPooling2D('pool5', 2, 2, border_mode='valid'))
    net.add(layer.Flatten('flat'))
    net.add(layer.Dropout('drop_flat', 0.5, engine='cudnn'))
    net.add(layer.Dense('ip1', 512))
    net.add(layer.BatchNormalization('batchnorm_ip1'))
    net.add(layer.Activation('relu_ip1'))
    net.add(layer.Dropout('drop_ip2', 0.5, engine='cudnn'))
    net.add(layer.Dense('ip2', 10))
    return net
