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

from singa import tensor
from singa.tensor import Tensor
from singa import autograd
from singa import optimizer
import numpy as np
from singa import sonnx

autograd.training = True
np.random.seed(0)
data = np.random.randn(4,3).astype(np.float32)
label = np.random.randint(0,2,(4)).astype(int)
print(label)
print(data.shape,label.shape)

def to_categorical(y, num_classes):
    '''
    Converts a class vector (integers) to binary class matrix.
    Args
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    Return
        A binary matrix representation of the input.
    '''
    y = np.array(y, dtype='int')
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

label = to_categorical(label, 3).astype(np.float32)
print('train_data_shape:', data.shape)
print('train_label_shape:', label.shape)

inputs = Tensor(data=data)
target = Tensor(data=label)


a,model = sonnx.onnx_model_init(inputs,'pytorch.onnx')

sgd = optimizer.SGD(0.00)

# training process
for epoch in range(1):
    loss = sonnx.onnx_loss(a,model,target)
    if (epoch % 100 == 0):
        print('training loss = ', tensor.to_numpy(loss)[0])


