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

from singa import opt
# import opt
from singa import device
from singa import tensor

import argparse
import time
import numpy as np
from tqdm import trange


def train_resnet(DIST=True, graph=True, sequential=False, verbosity=0):

    # Define the hypermeters for the train_resnet
    niters = 100
    batch_size = 32
    sgd = opt.SGD(lr=0.1, momentum=0.9, weight_decay=1e-5)

    IMG_SIZE = 224

    # For distributed training, sequential has better throughput in the current version
    if DIST == True:
        sgd = opt.DistOpt(
            sgd)  # Need to make sure DistOpt is working for multiple GPUs/nodes
        world_size = sgd.world_size
        local_rank = sgd.local_rank
        global_rank = sgd.global_rank
        sequential = True
    else:
        local_rank = 0
        world_size = 1
        global_rank = 0
        sequential = False

    dev = device.create_cuda_gpu_on(local_rank)

    tx = tensor.Tensor((batch_size, 3, IMG_SIZE, IMG_SIZE), dev)
    ty = tensor.Tensor((batch_size,), dev, tensor.int32)
    x = np.random.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    y = np.random.randint(0, 1000, batch_size, dtype=np.int32)
    tx.copy_from_numpy(x)
    ty.copy_from_numpy(y)

    dev.SetVerbosity(verbosity)
    dev.SetSkipIteration(5)

    # Construct the model
    from model import resnet
    model = resnet.resnet50(num_channels=3, num_classes=1000)

    model.train()
    model.set_optimizer(sgd)
    model.compile([tx], is_train=True, use_graph=graph, sequential=sequential)

    # Train model
    dev.Sync()
    start = time.time()
    with trange(niters) as t:
        for _ in t:
            model(tx, ty, dist_option='fp32', spars=None)

    # Device synchronization for more accurate time measurements during the distributed training
    dev.Sync()
    end = time.time()
    titer = (end - start) / float(niters)
    throughput = float(niters * batch_size * world_size) / (end - start)
    if global_rank == 0:
        print("\nThroughput = {} per second".format(throughput), flush=True)
        print("TotalTime={}".format(end - start), flush=True)
        print("Total={}".format(titer), flush=True)
        dev.PrintTimeProfiling()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Throughput test using Resnet 50')
    parser.add_argument('--dist',
                        '--enable-dist',
                        default='False',
                        action='store_true',
                        help='enable distributed training',
                        dest='DIST')
    parser.add_argument('--no-graph',
                        '--disable-graph',
                        default='True',
                        action='store_false',
                        help='disable graph',
                        dest='graph')
    parser.add_argument('--verbosity',
                        '--log-verbosity',
                        default=0,
                        type=int,
                        help='logging verbosity',
                        dest='verbosity')

    args = parser.parse_args()

    train_resnet(DIST=args.DIST,
                 graph=args.graph,
                 sequential=False,
                 verbosity=args.verbosity)
