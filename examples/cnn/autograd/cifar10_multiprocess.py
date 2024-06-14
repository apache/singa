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

from resnet_cifar10 import *
import multiprocessing
import sys

if __name__ == '__main__':

    # Generate a NCCL ID to be used for collective communication
    nccl_id = singa.NcclIdHolder()

    # Configure the number of GPUs to be used
    world_size = int(sys.argv[1])

    # Testing the experimental partial-parameter update asynchronous training
    partial_update = True

    process = []
    for local_rank in range(0, world_size):
        process.append(
            multiprocessing.Process(target=train_cifar10,
                                    args=(True, local_rank, world_size, nccl_id,
                                          partial_update)))

    for p in process:
        p.start()
