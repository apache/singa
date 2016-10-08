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
https://github.com/soumith/convnet-benchmarks/blob/master/caffe/imagenet_winners/overfeat.prototxt
'''

import sys
import timeit
import numpy as np

from singa import device
from singa import layer
from singa import loss
from singa import metric
from singa import net as ffnet
from singa import tensor
from singa import optimizer
from singa.proto import core_pb2

iterations = 10
batch_size = 128
input_shape = (3, 231, 231)

def create_net(use_cpu = False, use_ocl = False):
    if use_cpu:
        layer.engine = 'singacpp'
    if use_ocl:
        layer.engine = 'singacl'

    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
	
    # Conv 1
    net.add(layer.Conv2D("conv1", 96, 11, 4, input_sample_shape=input_shape))
    net.add(layer.Activation("conv1/relu"	))
    net.add(layer.MaxPooling2D("pool1/2x2_s2", 2, 2, border_mode='valid'))
    
    # Conv 2
    net.add(layer.Conv2D("conv1/5x5_s1", 256, 5, 1))
    net.add(layer.Activation("conv2/relu"))
    net.add(layer.MaxPooling2D("pool2/2x2_s2", 2, 2, border_mode='valid'))
    
    # Conv 3
    net.add(layer.Conv2D("conv3/3x3_s1", 512, 3, 1, pad=1))
    net.add(layer.Activation("conv3/relu"))
    
    # Conv 4
    net.add(layer.Conv2D("conv4/3x3_s1", 1024, 3, 1, pad=1))
    net.add(layer.Activation("conv4/relu"))
    
    # Conv 5
    net.add(layer.Conv2D("conv5/3x3_s1", 1024, 3, 1, pad=1))
    net.add(layer.Activation("conv5/relu"))
    net.add(layer.MaxPooling2D("pool5/2x2_s2", 2, 2, border_mode='valid'))
    
    # L2 Norm -> Inner product
    net.add(layer.Flatten("flat"))
    net.add(layer.Dense("fc6", 3072))
    net.add(layer.Activation("fc6/relu6"))
    
    net.add(layer.Dense("fc7", 4096))
    net.add(layer.Activation("fc7/relu7"))
    
    net.add(layer.Dense("fc8", 1000))

    for (val, spec) in zip(net.param_values(), net.param_specs()):
        filler = spec.filler
        if filler.type == 'gaussian':
            val.gaussian(filler.mean, filler.std)
        else:
            val.set_value(0)
        print spec.name, filler.type, val.l1()

    return net

# Time forward, backward, parameter update, per layer (1x forward, 1x backward)
def train(net, dev):
    tx = tensor.Tensor((batch_size,) + input_shape, dev)
    ty = tensor.Tensor((batch_size,), dev) # Should be integers, but CUDA with int tensor is not supported yet
    tx.gaussian(1.0, 0.5)
    ty.set_value(0.0)

    opt = optimizer.SGD(momentum=0.9)
    idx = np.arange(tx.shape[0], dtype = np.int32)
    loss = 0.0
    acc = 0.0
    
    train_time = 0.0
    update_time = 0.0
    for b in range(iterations):
        
        t0 = timeit.default_timer()
        grads, (l, a) = net.train(tx, ty)
        t1 = timeit.default_timer()
        t1 -= t0
        train_time += t1
        
        loss += l
        acc += a
        
        t2 = timeit.default_timer()
        for (s, p, g) in zip(net.param_names(), net.param_values(), grads):
            opt.apply_with_lr(0, 0.01, g, p, str(s), b)
        t3 = timeit.default_timer()
        t3 -= t2
        update_time += t3
                
        print("Iteration {}: Train: {}, Update: {}".format(b, t1, t3))
    
    print("Total iterations: {}".format(iterations))
    print("Average training time: {}".format(train_time/iterations))
    print("Average update time: {}".format(update_time/iterations))
        
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Pass in one argument of 'cpu', 'cuda', or 'opencl'.")
        quit()
    
    system = sys.argv[1]
    print("Running on {}.".format(system))
    
    if system == 'cpu':
        net = create_net(True, False)
        dev = device.get_default_device()
    elif system == 'cuda':
        net = create_net(False, False)
        dev = device.create_cuda_gpu()
    elif system == 'opencl':
        net = create_net(False, True)
        dev = device.create_opencl_device()

    net.to_device(dev)
    train(net, dev)
