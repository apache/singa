#
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
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

# the code is modified from
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

from singa import autograd
from singa import tensor
from singa import device
from singa import opt
#from singa import dist_opt

import numpy as np
from tqdm import trange

if __name__ == "__main__":
    sgd = opt.SGD(lr=0.1, momentum=0.9, weight_decay=1e-5)
    sgd = opt.DistOpt(sgd)

    from resnet import resnet50
    model = resnet50()

    print("Start intialization............")

    #sgd = dist_opt.Dist_SGD(lr=0.1)
    dev = device.create_cuda_gpu_on(sgd.rank_in_local)
    # dev = device.create_cuda_gpu()
    niters = 100
    batch_size = 32
    IMG_SIZE = 224

    tx = tensor.Tensor((batch_size, 3, IMG_SIZE, IMG_SIZE), dev)
    ty = tensor.Tensor((batch_size,), dev, tensor.int32)
    autograd.training = True
    x = np.random.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    y = np.random.randint(0, 1000, batch_size, dtype=np.int32)
    tx.copy_from_numpy(x)
    ty.copy_from_numpy(y)

    import time

    start = time.time()
    fd = 0
    softmax = 0
    update = 0
    with trange(niters) as t:
        for b in t:
            tick = time.time()
            x = model(tx)
            fd += time.time() - tick
            tick = time.time()
            loss = autograd.softmax_cross_entropy(x, ty)
            softmax += time.time() - tick
            for p, g in autograd.backward(loss):
                # print(p.shape, g.shape)
                tick = time.time()
                sgd.update(p, g)
                update += time.time() - tick
                # pass

    end = time.time()
    throughput = sgd.world_size * niters * batch_size / (end - start)
    print("Throughput = {} per second".format(throughput))
    titer = (end - start) / niters
    tforward = fd / niters
    tsoftmax = softmax / niters
    tbackward = titer - tforward - tsoftmax
    tsgd = update / niters
    print("Total={}, forward={}, softmax={}, backward={}, sgd={}".format(
        titer, tforward, tsoftmax, tbackward, tsgd))
