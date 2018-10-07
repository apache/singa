# singa.autograd

This part will present an overview of how autograd works and give a simple example of neuron network which implemented by using autograd API. 
## Autograd Mechanics
To get clear about how autograd system works, we should understand three important concepts in this system, they are `singa.tensor.Tensor` , `singa.autograd.Operation`, and `singa.autograd.Layer`.  For briefness, these three classes will be denoted as `tensor`, `operation`, and `layer`.
### Tensor
The class `tensor` has three attributes which are important in autograd system, they are `.creator`, `.requires_grad`, and `.stores_grad`.
-  `tensor.creator` should be an `operation` object. It records the particular `operation` which generates the `tensor ` itself.
-  `.requires_grad` and `.stores_grad` are both boolean indicator. These two attributes record whether a `tensor` needs gradients and whether gradients of a  `tensor` need to be stored when do backpropagation. It should be noted that if `.stores_grad` is true, then `.requires_grad` must be true， not vice versa.
### Operation
`operation` is the fundamental element of computation graph. A `operation` takes one or more `tensor` as input, and then output one or more `tensor`. The class `operation` has four important member functions:
 1.  `._do_forward()` Inputs and outputs of this function are both  `tensor`. This function will be invoked when the `operation` is called. So this function can be considered being controlled indirectly by users when they define their networks. This function mainly do two things: 
        - record source of the `operaiton`. Those inputs `tensor` contain their `creator` information, which are the source `operation` of current operation. Current `operation` keeps those information in the attribute `.src`. The designed autograd engine can control backward flow according to `operation.src`.
        - do calculation by calling member function `.forward()`, which will be introduced later.
 2.  `._do_backward()` Unlike `._do_forward()` is controlled by users to some extent, `._do_backward()` is in the charge of autograd engine. This kind of function don't take `tensor` as input, but instead, takes `tensor.data`. The effect of this function is to do calculation, by invoking `.backward()`
 4. `.forward()` and `.backward()` These two functions take `tensor.data` as inputs, and output `Ctensor`, which is the same type with `tensor.data`. To add a specific operation, subclass `operation` should implement their own `.forward()` and `.backward()`.
### Layer
For those operations contain parameters like weights which are `tensor`  requiring update, we package them into a new class, `layer`. Users should initialize a `layer` before invoking it.
When a `layer` is called, it will sent inputs `tensor` together with parameter `tensor` to corresponding operation, constructing the computation graph. We find that although a `layer` is called, there will be one more `operation` in computation graph rather than a `layer`.
## Python API
## Example
The following codes implement a Xception Net using autograd API. They can be found in source code of SINGA at 
  `incubator-singa/examples/autograd/xceptionnet.py`
### 1.  Import packages
```
from singa import autograd
from singa import tensor
from singa import device
from singa import opt

import numpy as np
from tqdm import trange
```
### 2. Create model
```
class Block(autograd.Layer):

    def __init__(self, in_filters, out_filters, reps, strides=1, padding=0, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = autograd.Conv2d(in_filters, out_filters,
                                        1, stride=strides, padding=padding, bias=False)
            self.skipbn = autograd.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.layers = []

        filters = in_filters
        if grow_first:
            self.layers.append(autograd.ReLU())
            self.layers.append(autograd.SeparableConv2d(in_filters, out_filters,
                                                        3, stride=1, padding=1, bias=False))
            self.layers.append(autograd.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            self.layers.append(autograd.ReLU())
            self.layers.append(autograd.SeparableConv2d(filters, filters,
                                                        3, stride=1, padding=1, bias=False))
            self.layers.append(autograd.BatchNorm2d(filters))

        if not grow_first:
            self.layers.append(autograd.ReLU())
            self.layers.append(autograd.SeparableConv2d(in_filters, out_filters,
                                                        3, stride=1, padding=1, bias=False))
            self.layers.append(autograd.BatchNorm2d(out_filters))

        if not start_with_relu:
            self.layers = self.layers[1:]
        else:
            self.layers[0] = autograd.ReLU()

        if strides != 1:
            self.layers.append(autograd.MaxPool2d(3, strides, padding + 1))

    def __call__(self, x):
        y = self.layers[0](x)
        for layer in self.layers[1:]:
            if isinstance(y, tuple):
                y = y[0]
            y = layer(y)

        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x
        y = autograd.add(y, skip)
        return y


class Xception(autograd.Layer):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = autograd.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = autograd.BatchNorm2d(32)

        self.conv2 = autograd.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = autograd.BatchNorm2d(64)

        self.block1 = Block(
            64, 128, 2, 2, padding=0, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, padding=0, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, padding=0, start_with_relu=True, grow_first=True)

        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = autograd.SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = autograd.BatchNorm2d(1536)

        # do relu here
        self.conv4 = autograd.SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = autograd.BatchNorm2d(2048)

        self.globalpooling = autograd.MaxPool2d(10, 1)
        self.fc = autograd.Linear(2048, num_classes)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = autograd.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = autograd.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = autograd.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = autograd.relu(features)
        x = self.globalpooling(x)
        x = autograd.flatten(x)
        x = self.fc(x)
        return x

    def __call__(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x
```

We can create a Xception Net by the following command:

`model = Xception(num_classes=1000)`

### 3. Sample data
```
IMG_SIZE = 299
tx = tensor.Tensor((batch_size, 3, IMG_SIZE, IMG_SIZE), dev)
ty = tensor.Tensor((batch_size,), dev, tensor.int32)
x = np.random.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
y = np.random.randint(0, 1000, batch_size, dtype=np.int32)
tx.copy_from_numpy(x)
ty.copy_from_numpy(y)
```

### 4. Set learning parameters and create optimizer
```
niters = 20
batch_size = 16
sgd = opt.SGD(lr=0.1, momentum=0.9, weight_decay=1e-5)
```
### 5. Train model
Set `autograd.training` as true:
`autograd.training = True`

Then start training:
```
with trange(niters) as t:
        for b in t:
            x = model(tx)
            loss = autograd.softmax_cross_entropy(x, ty)
            for p, g in autograd.backward(loss):
                sgd.update(p, g)
```
 



