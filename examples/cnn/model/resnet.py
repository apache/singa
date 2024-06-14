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

from singa import layer
from singa import model


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(
        in_planes,
        out_planes,
        3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(layer.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = layer.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = layer.BatchNorm2d(planes)
        self.relu1 = layer.ReLU()
        self.add = layer.Add()
        self.relu2 = layer.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.add(out, residual)
        out = self.relu2(out)

        return out


class Bottleneck(layer.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = layer.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = layer.BatchNorm2d(planes)
        self.relu1 = layer.ReLU()
        self.conv2 = layer.Conv2d(planes,
                                  planes,
                                  3,
                                  stride=stride,
                                  padding=1,
                                  bias=False)
        self.bn2 = layer.BatchNorm2d(planes)
        self.relu2 = layer.ReLU()
        self.conv3 = layer.Conv2d(planes,
                                  planes * self.expansion,
                                  1,
                                  bias=False)
        self.bn3 = layer.BatchNorm2d(planes * self.expansion)

        self.add = layer.Add()
        self.relu3 = layer.ReLU()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.add(out, residual)
        out = self.relu3(out)

        return out


__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]


class ResNet(model.Model):

    def __init__(self, block, layers, num_classes=10, num_channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.input_size = 224
        self.dimension = 4
        self.conv1 = layer.Conv2d(num_channels,
                                  64,
                                  7,
                                  stride=2,
                                  padding=3,
                                  bias=False)
        self.bn1 = layer.BatchNorm2d(64)
        self.relu = layer.ReLU()
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1, layers1 = self._make_layer(block, 64, layers[0])
        self.layer2, layers2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3, layers3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4, layers4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = layer.AvgPool2d(7, stride=1)
        self.flatten = layer.Flatten()
        self.fc = layer.Linear(num_classes)
        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()

        self.register_layers(*layers1, *layers2, *layers3, *layers4)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv = layer.Conv2d(
                self.inplanes,
                planes * block.expansion,
                1,
                stride=stride,
                bias=False,
            )
            bn = layer.BatchNorm2d(planes * block.expansion)

            def _downsample(x):
                return bn(conv(x))

            downsample = _downsample

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        def forward(x):
            for layer in layers:
                x = layer(x)
            return x

        return forward, layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

    def train_one_batch(self, x, y, dist_option, spars):
        out = self.forward(x)
        loss = self.softmax_cross_entropy(out, y)

        if dist_option == 'plain':
            self.optimizer(loss)
        elif dist_option == 'half':
            self.optimizer.backward_and_update_half(loss)
        elif dist_option == 'partialUpdate':
            self.optimizer.backward_and_partial_update(loss)
        elif dist_option == 'sparseTopK':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=True,
                                                      spars=spars)
        elif dist_option == 'sparseThreshold':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=False,
                                                      spars=spars)
        return out, loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.

    Returns:
        The created ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.

    Returns:
        The created ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.

    Returns:
        The created ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.

    Returns:
        The created ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.

    Returns:
        The created ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model


__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]
