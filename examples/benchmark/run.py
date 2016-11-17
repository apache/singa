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
https://github.com/soumith/convnet-benchmarks/blob/master/caffe/imagenet_winners/alexnet.prototxt
'''

import sys
import numpy as np

from singa import device
from singa import tensor
from singa import optimizer

iterations = 10
batch_size = 128
input_shape = (3, 224, 224)

# Time forward, backward, parameter update, per layer (1x forward, 1x backward)
def train(net, dev):
    tx = tensor.Tensor((batch_size,) + input_shape, dev)
    ty = tensor.Tensor((batch_size,), dev)
    tx.gaussian(1.0, 0.5)
    ty.set_value(0.0)

    opt = optimizer.SGD(momentum=0.9)
    idx = np.arange(tx.shape[0], dtype = np.int32)
    loss = 0.0
    acc = 0.0

    train_time = 0.0
    update_time = 0.0
    net.start_benchmark()
    update = 0
    for b in range(iterations):
        grads, (l, a) = net.train_benchmark(tx, ty)
        t1 = timer()
        for (s, p, g) in zip(net.param_names(), net.param_values(), grads):
            opt.apply_with_lr(0, 0.01, g, p, str(s), b)
        update += timer() - t1
    t, fp, bp, fps, bps = net.stop_benchmark(iterations)

    print "Total iterations = %d" % iterations
    print "Average training time per iteration = %.4f" % t
    print "Average forward time per iteration = %.4f" % fp
    print "Average backward time per iteration = %.4f" % bp
    print "Average udpate time per iteration = %.4f" % (update / iterations)
    for (k, v) in fps:
        print "Forward time for %10s = %.4f" % (k, v)
    for (k, v) in bps:
        print "Backward time for %10s = %.4f" % (k, v)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark SINGA by running'
                                     'AlexNet/VGG/Overfeat with CPP/CUDA/Opencl')
    parser.add_argument('net', choices=['vgg', 'alexnet', 'overfeat'],
                        default='alexnet')
    parser.add_argument('device', choices=['cpp', 'cuda', 'opencl'],
                        default='cuda')
    args = parser.parse_args()
    if args.net == 'vgg':
        import vgg as model
    elif args.net == 'alexnet':
        import alexnet as model
    else:
        assert args.net == 'overfeat', 'Wrong net type:' + args.net
        import overfeat as model

    use_cpu = False,
    use_opencl = False

    if args.device == 'cpu':
        use_cpu = True
        dev = device.get_default_device()
    elif args.device == 'cuda':
        dev = device.create_cuda_gpu()
    else:
        assert args.device == 'opencl', 'Wrong lang: ' + args.device
        use_opencl = True
        dev = device.create_opencl_device()
    net = model.create_net(input_shape, use_cpu, use_opencl)
    net.to_device(dev)
    train(net, dev)
