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

#!/usr/bin/env python -W ignore::DeprecationWarning

# resnet
mpiexec -np 8 python train_mpi.py resnet mnist -l 0.015 -b 32
mpiexec -np 8 python train_mpi.py resnet cifar10 -l 0.015 -b 32
mpiexec -np 8 python train_mpi.py resnet cifar100 -l 0.015 -b 32

# cnn
mpiexec -np 8 python train_mpi.py cnn mnist -l 0.015 -b 32
mpiexec -np 8 python train_mpi.py cnn cifar10 -l 0.015 -b 32
mpiexec -np 8 python train_mpi.py cnn cifar100 -l 0.015 -b 32

# mlp
mpiexec -np 8 python train_mpi.py mlp mnist -l 0.015 -b 32
mpiexec -np 8 python train_mpi.py mlp cifar10 -l 0.015 -b 32
mpiexec -np 8 python train_mpi.py mlp cifar100 -l 0.015 -b 32

# alexnet
mpiexec -np 8 python train_mpi.py alexnet mnist -l 0.015 -b 32
mpiexec -np 8 python train_mpi.py alexnet cifar10 -l 0.015 -b 32
mpiexec -np 8 python train_mpi.py alexnet cifar100 -l 0.015 -b 32

# xceptionnet
mpiexec -np 8 python train_mpi.py xceptionnet mnist -l 0.015 -b 32
mpiexec -np 8 python train_mpi.py xceptionnet cifar10 -l 0.015 -b 32
