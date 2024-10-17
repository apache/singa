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

from singa import singa_wrap as singa
from singa import opt
from singa import tensor
import argparse
import train_cnn
import multiprocessing

singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}


def run(args, local_rank, world_size, nccl_id):
    sgd = opt.SGD(lr=args.lr,
                  momentum=0.9,
                  weight_decay=1e-5,
                  dtype=singa_dtype[args.precision])
    sgd = opt.DistOpt(sgd,
                      nccl_id=nccl_id,
                      local_rank=local_rank,
                      world_size=world_size)
    train_cnn.run(sgd.global_rank, sgd.world_size, sgd.local_rank,
                  args.max_epoch, args.batch_size, args.model, args.data, sgd,
                  args.graph, args.verbosity, args.dist_option, args.spars,
                  args.precision)


if __name__ == '__main__':
    # Use argparse to get command config: max_epoch, model, data, etc., for single gpu training
    parser = argparse.ArgumentParser(
        description='Training using the autograd and graph.')
    parser.add_argument('model',
                        choices=['resnet', 'xceptionnet', 'cnn', 'mlp'],
                        default='cnn')
    parser.add_argument('data',
                        choices=['cifar10', 'cifar100', 'mnist'],
                        default='mnist')
    parser.add_argument('-p',
                        choices=['float32', 'float16'],
                        default='float32',
                        dest='precision')
    parser.add_argument('-m',
                        '--max-epoch',
                        default=10,
                        type=int,
                        help='maximum epochs',
                        dest='max_epoch')
    parser.add_argument('-b',
                        '--batch-size',
                        default=64,
                        type=int,
                        help='batch size',
                        dest='batch_size')
    parser.add_argument('-l',
                        '--learning-rate',
                        default=0.005,
                        type=float,
                        help='initial learning rate',
                        dest='lr')
    parser.add_argument('-w',
                        '--world-size',
                        default=2,
                        type=int,
                        help='number of gpus to be used',
                        dest='world_size')
    parser.add_argument(
        '-d',
        '--dist-option',
        default='plain',
        choices=[
            'plain', 'half', 'partialUpdate', 'sparseTopK', 'sparseThreshold'
        ],
        help='distibuted training options',
        dest='dist_option')  # currently partialUpdate support graph=False only
    parser.add_argument(
        '-s',
        '--sparsification',
        default='0.05',
        type=float,
        help='the sparsity parameter used for sparsification, between 0 to 1',
        dest='spars')
    parser.add_argument('-g',
                        '--disable-graph',
                        default='True',
                        action='store_false',
                        help='disable graph',
                        dest='graph')
    parser.add_argument('-v',
                        '--log-verbosity',
                        default=0,
                        type=int,
                        help='logging verbosity',
                        dest='verbosity')

    args = parser.parse_args()

    # Generate a NCCL ID to be used for collective communication
    nccl_id = singa.NcclIdHolder()

    process = []
    for local_rank in range(0, args.world_size):
        process.append(
            multiprocessing.Process(target=run,
                                    args=(args, local_rank, args.world_size,
                                          nccl_id)))

    for p in process:
        p.start()
