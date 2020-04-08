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
import argparse
import train

if __name__ == '__main__':
    # use argparse to get command config: max_epoch, model, data, etc. for single gpu training
    parser = argparse.ArgumentParser(
        description='Training using the autograd and graph.')
    parser.add_argument('model',
                        choices=['resnet', 'xceptionnet', 'cnn', 'mlp'],
                        default='cnn')
    parser.add_argument('data', choices=['cifar10', 'cifar100', 'mnist'], default='mnist')
    parser.add_argument('--epoch',
                        '--max-epoch',
                        default=10,
                        type=int,
                        help='maximum epochs',
                        dest='max_epoch')
    parser.add_argument('--bs',
                        '--batch-size',
                        default=64,
                        type=int,
                        help='batch size',
                        dest='batch_size')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.005,
                        type=float,
                        help='initial learning rate',
                        dest='lr')
    parser.add_argument('--op',
                        '--option',
                        default='fp32',
                        choices=['fp32','fp16','partialUpdate','sparseTopK','sparseThreshold'],
                        help='distibuted training options',
                        dest='dist_option')  # currently partialUpdate support graph=False only 
    parser.add_argument('--spars',
                        '--sparsification',
                        default='0.05',
                        type=float,
                        help='the sparsity parameter used for sparsification, between 0 to 1',
                        dest='spars')
    parser.add_argument('--no-graph',
                        '--disable-graph',
                        default='True',
                        action='store_false',
                        help='disable graph',
                        dest='graph')

    args = parser.parse_args()

    sgd = opt.SGD(lr=args.lr, momentum=0.9, weight_decay=1e-5)
    sgd = opt.DistOpt(sgd)

    train.run(sgd.global_rank, sgd.world_size, sgd.local_rank, args.max_epoch,
              args.batch_size, args.model, args.data, sgd, args.graph, args.dist_option, args.spars)
