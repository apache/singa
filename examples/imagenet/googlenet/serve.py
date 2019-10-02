# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
''' This model is created following Caffe implementation of GoogleNet
https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/
'''
from __future__ import print_function
from builtins import zip
from builtins import str
import sys
import time
import numpy as np
import traceback
from argparse import ArgumentParser
from scipy.misc import imread
import numpy as np

from singa.layer import Layer, Conv2D, Activation, MaxPooling2D, AvgPooling2D,\
        Split, Concat, LRN, Dropout, Flatten, Dense
from singa import layer
from singa import net as ffnet
from singa import device
from singa import tensor

from rafiki.agent import Agent, MsgType


def add_to_tuple(x):
    '''return a tuple with the last two values incremented by 1'''
    if len(x) == 3:
        return (x[0], x[1] + 1, x[2] + 1)
    else:
        return (x[0], x[1], x[2] + 1, x[3] + 1)

class EndPadding(Layer):
    '''Pad the end of the spatial axis with 1 row and 1 column of zeros.

    This layer is inserted before the pooling layers outside the inception
    block. We need such a layer because Caffe (ceil) and cuDNN (floor) have
    different rounding strategies for the pooling layer.
    http://joelouismarino.github.io/blog_posts/blog_googlenet_keras.html
    '''
    def __init__(self, name, input_sample_shape=None):
        super(EndPadding, self).__init__(name)
        if input_sample_shape is not None:
            assert len(input_sample_shape) == 3, 'input must has 4 dims'
            self.output_sample_shape = add_to_tuple(input_sample_shape)

    def get_output_sample_shape(self):
        return self.output_sample_shape

    def setup(self, input_sample_shape):
        assert len(input_sample_shape) == 3, 'input must has 4 dims'
        self.output_sample_shape = add_to_tuple(input_sample_shape)
        self.has_setup = True

    def forward(self, flag, x):
        '''pad zeros'''
        tmp = tensor.to_numpy(x)
        shape = add_to_tuple(x.shape)
        ret = np.zeros(shape)
        ret[:,:,:-1, :-1] = tmp
        y = tensor.from_numpy(ret)
        y.to_device(x.device)
        return y

    def backward(self, falg, dy):
        '''remove paddings'''
        tmp = tensor.to_numpy(dy)
        dx = tensor.from_numpy(tmp[:,:,:-1,:-1])
        dx.to_device(dy.device)
        return dx, []

# b_specs = {'init': 'constant', 'value': 0, 'lr_mult': 2, 'decay_mult': 0}

def conv(net, src, name, num, kernel, stride=1, pad=0, suffix=''):
    net.add(Conv2D('%s/%s' % (name, suffix), num, kernel, stride, pad=pad), src)
    return net.add(Activation('%s/relue_%s' % (name, suffix)))

def pool(net, src, name, kernel, stride):
    net.add(EndPadding('%s/pad' % name), src)
    ret = net.add(MaxPooling2D('%s' % name, 3, 2, pad=0))
    return ret

def inception(net, src, name, nb1x1, nb3x3r, nb3x3, nb5x5r, nb5x5, nbproj):
    split = net.add(Split('%s/split' % name, 4), src)

    c1x1 = conv(net, split, name, nb1x1, 1, suffix='1x1')

    c3x3r = conv(net, split, name, nb3x3r, 1, suffix='3x3_reduce')
    c3x3 = conv(net, c3x3r, name, nb3x3, 3, pad=1, suffix='3x3')

    c5x5r = conv(net, split, name, nb5x5r, 1, suffix='5x5_reduce')
    c5x5 = conv(net, c5x5r, name, nb5x5, 5, pad=2, suffix='5x5')

    pool = net.add(MaxPooling2D('%s/pool' % name, 3, 1, pad=1), split)
    cproj = conv(net, pool, name, nbproj, 1, suffix='pool_proj')

    return net.add(Concat('%s/output' % name, 1), [c1x1, c3x3, c5x5, cproj])


def create_net(shape, weight_path='bvlc_googlenet.pickle'):
    net = ffnet.FeedForwardNet()
    net.add(Conv2D('conv1/7x7_s2', 64, 7, 2, pad=3, input_sample_shape=shape))
    c1 = net.add(Activation('conv1/relu_7x7'))
    pool(net, c1, 'pool1/3x3_s2', 3, 2)
    norm1 = net.add(LRN('pool1/norm1', 5, 0.0001, 0.75))
    c3x3r = conv(net, norm1 , 'conv2', 64, 1, suffix='3x3_reduce')
    conv(net, c3x3r, 'conv2', 192, 3, pad=1, suffix='3x3')
    norm2 = net.add(LRN('conv2/norm2', 5, 0.0001, 0.75))
    pool2 = pool(net, norm2, 'pool2/3x3_s2', 3, 2)

    i3a=inception(net, pool2, 'inception_3a', 64, 96, 128, 16, 32, 32)
    i3b=inception(net, i3a, 'inception_3b', 128, 128, 192, 32, 96, 64)
    pool3=pool(net, i3b, 'pool3/3x3_s2', 3, 2)
    i4a=inception(net, pool3, 'inception_4a', 192, 96, 208, 16, 48, 64)
    i4b=inception(net, i4a, 'inception_4b', 160, 112, 224, 24, 64, 64)
    i4c=inception(net, i4b, 'inception_4c', 128, 128, 256, 24, 64, 64)
    i4d=inception(net, i4c, 'inception_4d', 112, 144, 288, 32, 64, 64)
    i4e=inception(net, i4d, 'inception_4e', 256, 160, 320, 32, 128, 128)
    pool4=pool(net, i4e,'pool4/3x3_s2', 3, 2)
    i5a=inception(net, pool4, 'inception_5a', 256, 160, 320, 32, 128, 128)
    inception(net, i5a, 'inception_5b', 384, 192, 384, 48, 128, 128)
    net.add(AvgPooling2D('pool5/7x7_s1', 7, 1, pad=0))
    net.add(Dropout('drop', 0.4))
    net.add(Flatten('flat'))
    net.add(Dense('loss3/classifier', 1000))
    # prob=net.add(Softmax('softmax'))

    net.load(weight_path, use_pickle=True)
    print('total num of params %d' % (len(net.param_names())))
    # SINGA and Caffe have different layout for the weight matrix of the dense
    # layer
    for key, val in zip(net.param_names(), net.param_values()):
        # print key
        if key == 'loss3/classifier_weight' or key == 'loss3/classifier/weight':
            tmp = tensor.to_numpy(val)
            tmp = tmp.reshape(tmp.shape[::-1])
            val.copy_from_numpy(np.transpose(tmp))
    return net


def serve(agent, use_cpu, parameter_file, topk=5):
    if use_cpu:
        print('running with cpu')
        dev = device.get_default_device()
        layer.engine = 'singacpp'
    else:
        print("runing with gpu")
        dev = device.create_cuda_gpu()

    print('Start intialization............')
    net = create_net((3, 224, 224), parameter_file)
    net.to_device(dev)
    print('End intialization............')

    labels = np.loadtxt('synset_words.txt', str, delimiter='\t ')
    while True:
        key, val = agent.pull()
        if key is None:
            time.sleep(0.1)
            continue
        msg_type = MsgType.parse(key)
        if msg_type.is_request():
            try:
                response = ""
                img = imread(val['image'], mode='RGB').astype(np.float32)
                height,width = img.shape[:2]
                img[:, :, 0] -= 123.68
                img[:, :, 1] -= 116.779
                img[:, :, 2] -= 103.939
                img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
                img = img.transpose((2, 0, 1))
                img = img[:, (height-224)//2:(height+224)//2,\
                          (width-224)//2:(width+224)//2]
                images = np.expand_dims(img, axis=0)

                x = tensor.from_numpy(images.astype(np.float32))
                x.to_device(dev)
                y = net.predict(x)
                prob = np.average(tensor.to_numpy(y), 0)
                # sort and reverse
                idx = np.argsort(-prob)[0:topk]
                for i in idx:
                    response += "%s:%s<br/>" % (labels[i], prob[i])
            except Exception:
                traceback.print_exc()
                response = "Sorry, system error during prediction."
            except SystemExit:
                traceback.print_exc()
                response = "Sorry, error triggered sys.exit() during prediction."
            agent.push(MsgType.kResponse, response)
        elif MsgType.kCommandStop.equal(msg_type):
                print('get stop command')
                agent.push(MsgType.kStatus, "success")
                break
        else:
            print('get unsupported message %s' % str(msg_type))
            agent.push(MsgType.kStatus, "Unknown command")
            break
        # while loop
    print("server stop")


def main():
    try:
        # Setup argument parser
        parser = ArgumentParser(description="GoogleNet for image classification")
        parser.add_argument("-p", "--port", default=9999, help="listen port")
        parser.add_argument("-C", "--use_cpu", action="store_true")
        parser.add_argument("--parameter_file", default="bvlc_googlenet.pickle",
                help="relative path")

        # Process arguments
        args = parser.parse_args()
        port = args.port

        # start to train
        agent = Agent(port)
        serve(agent, args.use_cpu, args.parameter_file)
        agent.stop()

    except SystemExit:
        return
    except Exception:
        traceback.print_exc()
        sys.stderr.write("  for help use --help \n\n")
        return 2


if __name__ == '__main__':
    main()
