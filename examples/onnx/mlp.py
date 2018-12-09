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
from singa import *
from singa import sonnx
import onnx

import numpy as np
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

w0 = Tensor(shape=(3, 3), requires_grad=True, stores_grad=True)
w0.gaussian(0.0, 0.1)
b0 = Tensor(shape=(1, 3), requires_grad=True, stores_grad=True)
b0.set_value(0.0)

w1 = Tensor(shape=(3, 3), requires_grad=True, stores_grad=True)
w1.gaussian(0.0, 0.1)
b1 = Tensor(shape=(1, 3), requires_grad=True, stores_grad=True)
b1.set_value(0.0)

w2 = Tensor(shape=(3, 3), requires_grad=True, stores_grad=True)
w2.gaussian(0.0, 0.1)
b2 = Tensor(shape=(1, 3), requires_grad=True, stores_grad=True)
b2.set_value(0.0)



sgd = optimizer.SGD(0.00)

# training process
for i in range(1):
    x = autograd.matmul(inputs, w0)
    x = autograd.add_bias(x, b0)
    x = autograd.relu(x)
    x2 = autograd.matmul(x, w2)
    x2 = autograd.add_bias(x2, b2)
    x1 = autograd.matmul(x, w1)
    x1 = autograd.add_bias(x1, b1)
    x3 = autograd.add(x1, x2)
    x3 = autograd.softmax(x3)
    loss = autograd.cross_entropy(x3, target)
    gradient = autograd.backward(loss)
    for p, gp in gradient:
        sgd.apply(0, gp, p, '')
    if (i % 100 == 0):
        print('training loss = ', tensor.to_numpy(loss)[0])


model=sonnx.get_onnx_model(loss,inputs,target)

onnx.save(model, 'mlp.onnx')

