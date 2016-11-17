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
''' This model is created following the structure from
https://github.com/soumith/convnet-benchmarks/blob/master/caffe/imagenet_winners/vgg_a.prototxt
'''

from singa import layer
from singa import loss
from singa import metric
from singa import net as ffnet


def create_net(input_shape, use_cpu=False, use_ocl=False):
    if use_cpu:
        layer.engine = 'singacpp'
    if use_ocl:
        layer.engine = 'singacl'

    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())

    net.add(layer.Conv2D("conv1/3x3_s1", 64, 3, 1, pad=1,
                         input_sample_shape=input_shape))
    net.add(layer.Activation("conv1/relu"))
    net.add(layer.MaxPooling2D("pool1/2x2_s2", 2, 2, border_mode='valid'))

    net.add(layer.Conv2D("conv2/3x3_s1", 128, 3, 1, pad=1))
    net.add(layer.Activation("conv2/relu"))
    net.add(layer.MaxPooling2D("pool2/2x2_s2", 2, 2, border_mode='valid'))

    net.add(layer.Conv2D("conv3/3x3_s1", 256, 3, 1, pad=1))
    net.add(layer.Activation("conv3/relu"))
    # No pooling layer here.

    net.add(layer.Conv2D("conv4/3x3_s1", 256, 3, 1, pad=1))
    net.add(layer.Activation("conv4/relu"))
    net.add(layer.MaxPooling2D("pool3/2x2_s2", 2, 2, border_mode='valid'))

    net.add(layer.Conv2D("conv5/3x3_s1", 512, 3, 1, pad=1))
    net.add(layer.Activation("conv5/relu"))
    # No pooling layer here.

    net.add(layer.Conv2D("conv6/3x3_s1", 512, 3, 1, pad=1))
    net.add(layer.Activation("conv6/relu"))
    net.add(layer.MaxPooling2D("pool4/2x2_s2", 2, 2, border_mode='valid'))

    net.add(layer.Conv2D("conv7/3x3_s1", 512, 3, 1, pad=1))
    net.add(layer.Activation("conv7/relu"))
    # No pooling layer here.

    net.add(layer.Conv2D("conv8/3x3_s1", 512, 3, 1, pad=1))
    net.add(layer.Activation("conv8/relu"))
    net.add(layer.MaxPooling2D("pool5/2x2_s2", 2, 2, border_mode='valid'))

    net.add(layer.Flatten('flat'))
    net.add(layer.Dense("fc6", 4096))
    net.add(layer.Dense("fc7", 4096))
    net.add(layer.Dense("fc8", 1000))

    for (val, spec) in zip(net.param_values(), net.param_specs()):
        filler = spec.filler
        if filler.type == 'gaussian':
            val.gaussian(filler.mean, filler.std)
        else:
            val.set_value(0)
        print spec.name, filler.type, val.l1()

    return net
