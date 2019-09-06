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

from singa import opt
from mnist_cnn import *
import multiprocessing

def data_partition(dataset_x, dataset_y, rank_in_global, world_size):
    data_per_rank = dataset_x.shape[0] // world_size
    idx_start = rank_in_global * data_per_rank
    idx_end = (rank_in_global + 1) * data_per_rank
    return dataset_x[idx_start: idx_end], dataset_y[idx_start: idx_end]

if __name__ == '__main__':

    # Generate a NCCL ID to be used for collective communication
    nccl_id = singa.NcclIdHolder()

    sgd = opt.SGD(lr=0.04, momentum=0.9, weight_decay=1e-5)    

    gpu_per_node = 8
    max_epoch = 10
    batch_size = 64

    process = []
    for gpu_num in range(0, gpu_per_node):        
        process.append(multiprocessing.Process(target=train_mnist_cnn, args=(sgd, max_epoch, batch_size, True, data_partition, gpu_num, gpu_per_node, nccl_id))) 

    for p in process:
        p.start()
