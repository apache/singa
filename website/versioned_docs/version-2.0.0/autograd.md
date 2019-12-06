---
id: version-2.0.0-autograd
title: Autograd in SINGA
original_id: autograd
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

There are two typical ways to implement autograd, via symbolic differentiation like [Theano](http://deeplearning.net/software/theano/index.html) or reverse differentiation like [Pytorch](https://pytorch.org/docs/stable/notes/autograd.html). Singa follows Pytorch way, which records the computation graph and apply the backward propagation automatically after forward propagation. The autograd algorithm is explained in details [here](https://pytorch.org/docs/stable/notes/autograd.html). We explain the relevant modules in Singa and give an example to illustrate the usage.

## Relevant Modules

There are three classes involved in autograd, namely `singa.tensor.Tensor`, `singa.autograd.Operation`, and `singa.autograd.Layer`. In the rest of this article, we use tensor, operation and layer to refer to an instance of the respective class.

### Tensor

Three attributes of Tensor are used by autograd,

- `.creator` is an `Operation` instance. It records the operation that generates the Tensor instance.
- `.requires_grad` is a boolean variable. It is used to indicate that the autograd algorithm needs to compute the gradient of the tensor (i.e., the owner). For example, during backpropagation, the gradients of the tensors for the weight matrix of a linear layer and the feature maps of a convolution layer (not the bottom layer) should be computed.
- `.stores_grad` is a boolean variable. It is used to indicate that the gradient of the owner tensor should be stored and output by the backward function. For example, the gradient of the feature maps is computed during backpropagation, but is not included in the output of the backward function.

Programmers can change `requires_grad` and `stores_grad` of a Tensor instance. For example, if later is set to True, the corresponding gradient is included in the output of the backward function. It should be noted that if `stores_grad` is True, then `requires_grad` must be true, not vice versa.

### Operation

It takes one or more `Tensor` instances as input, and then outputs one or more `Tensor` instances. For example, ReLU can be implemented as a specific Operation subclass. When an `Operation` instance is called (after instantiation), the following two steps are executed:

1. record the source operations, i.e., the `creator`s of the input tensors.
2. do calculation by calling member function `.forward()`

There are two member functions for forwarding and backwarding, i.e., `.forward()` and `.backward()`. They take `Tensor.data` as inputs (the type is `CTensor`), and output `Ctensor`s. To add a specific operation, subclass `operation` should implement their own `.forward()` and `.backward()`. The `backward()` function is called by the `backward()` function of autograd automatically during backward propogation to compute the gradients of inputs (according to the `require_grad` field).

### Layer

For those operations that require parameters, we package them into a new class, `Layer`. For example, convolution operation is wrapped into a convolution layer. `Layer` manages (stores) the parameters and calls the corresponding `Operation`s to implement the transformation.

## Examples

Multiple examples are provided in the [example folder](https://github.com/apache/singa/tree/master/examples/autograd). We explain two representative examples here.

### Operation only

The following codes implement a MLP model using only Operation instances (no Layer instances).

#### Import packages

```python
from singa.tensor import Tensor
from singa import autograd
from singa import opt
```

#### Create weight matrix and bias vector

The parameter tensors are created with both `requires_grad` and `stores_grad` set to `True`.

```python
w0 = Tensor(shape=(2, 3), requires_grad=True, stores_grad=True)
w0.gaussian(0.0, 0.1)
b0 = Tensor(shape=(1, 3), requires_grad=True, stores_grad=True)
b0.set_value(0.0)

w1 = Tensor(shape=(3, 2), requires_grad=True, stores_grad=True)
w1.gaussian(0.0, 0.1)
b1 = Tensor(shape=(1, 2), requires_grad=True, stores_grad=True)
b1.set_value(0.0)
```

#### Training

```python
inputs = Tensor(data=data)  # data matrix
target = Tensor(data=label) # label vector
autograd.training = True    # for training
sgd = opt.SGD(0.05)   # optimizer

for i in range(10):
    x = autograd.matmul(inputs, w0) # matrix multiplication
    x = autograd.add_bias(x, b0)    # add the bias vector
    x = autograd.relu(x)            # ReLU activation operation

    x = autograd.matmul(x, w1)
    x = autograd.add_bias(x, b1)

    loss = autograd.softmax_cross_entropy(x, target)

    for p, g in autograd.backward(loss):
        sgd.update(p, g)
```

### Operation + Layer

The following [example](https://github.com/apache/singa/blob/master/examples/autograd/mnist_cnn.py) implements a CNN model using layers provided by the autograd module.

#### Create the layers

```python
conv1 = autograd.Conv2d(1, 32, 3, padding=1, bias=False)
bn1 = autograd.BatchNorm2d(32)
pooling1 = autograd.MaxPool2d(3, 1, padding=1)
conv21 = autograd.Conv2d(32, 16, 3, padding=1)
conv22 = autograd.Conv2d(32, 16, 3, padding=1)
bn2 = autograd.BatchNorm2d(32)
linear = autograd.Linear(32 * 28 * 28, 10)
pooling2 = autograd.AvgPool2d(3, 1, padding=1)
```

#### Define the forward function

The operations in the forward pass will be recorded automatically for backward propagation.

```python
def forward(x, t):
    # x is the input data (a batch of images)
    # t the the label vector (a batch of integers)
    y = conv1(x)           # Conv layer
    y = autograd.relu(y)   # ReLU operation
    y = bn1(y)             # BN layer
    y = pooling1(y)        # Pooling Layer

    # two parallel convolution layers
    y1 = conv21(y)
    y2 = conv22(y)
    y = autograd.cat((y1, y2), 1)  # cat operation
    y = autograd.relu(y)           # ReLU operation
    y = bn2(y)
    y = pooling2(y)

    y = autograd.flatten(y)        # flatten operation
    y = linear(y)                  # Linear layer
    loss = autograd.softmax_cross_entropy(y, t)  # operation
    return loss, y
```

#### Training

```python
autograd.training = True
for epoch in range(epochs):
    for i in range(batch_number):
        inputs = tensor.Tensor(device=dev, data=x_train[
                               i * batch_sz:(1 + i) * batch_sz], stores_grad=False)
        targets = tensor.Tensor(device=dev, data=y_train[
                                i * batch_sz:(1 + i) * batch_sz], requires_grad=False, stores_grad=False)

        loss, y = forward(inputs, targets) # forward the net

        for p, gp in autograd.backward(loss):  # auto backward
            sgd.update(p, gp)
```
