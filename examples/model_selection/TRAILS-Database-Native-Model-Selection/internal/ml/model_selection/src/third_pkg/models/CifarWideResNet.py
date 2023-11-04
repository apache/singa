import torch
import torch.nn as nn
import torch.nn.functional as F
from .initialization import initialize_resnet


class WideBasicblock(nn.Module):
    def __init__(self, inplanes, planes, stride, dropout=False):
        super(WideBasicblock, self).__init__()

        self.bn_a = nn.BatchNorm2d(inplanes)
        self.conv_a = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.bn_b = nn.BatchNorm2d(planes)
        if dropout:
            self.dropout = nn.Dropout2d(p=0.5, inplace=True)
        else:
            self.dropout = None
        self.conv_b = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if inplanes != planes:
            self.downsample = nn.Conv2d(
                inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False
            )
        else:
            self.downsample = None

    def forward(self, x):

        basicblock = self.bn_a(x)
        basicblock = F.relu(basicblock)
        basicblock = self.conv_a(basicblock)

        basicblock = self.bn_b(basicblock)
        basicblock = F.relu(basicblock)
        if self.dropout is not None:
            basicblock = self.dropout(basicblock)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            x = self.downsample(x)

        return x + basicblock


class CifarWideResNet(nn.Module):
    """
    ResNet optimized for the Cifar dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, depth, widen_factor, num_classes, dropout):
        super(CifarWideResNet, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 4) % 6 == 0, "depth should be one of 20, 32, 44, 56, 110"
        layer_blocks = (depth - 4) // 6
        print(
            "CifarPreResNet : Depth : {} , Layers for each block : {}".format(
                depth, layer_blocks
            )
        )

        self.num_classes = num_classes
        self.dropout = dropout
        self.conv_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.message = "Wide ResNet : depth={:}, widen_factor={:}, class={:}".format(
            depth, widen_factor, num_classes
        )
        self.inplanes = 16
        self.stage_1 = self._make_layer(
            WideBasicblock, 16 * widen_factor, layer_blocks, 1
        )
        self.stage_2 = self._make_layer(
            WideBasicblock, 32 * widen_factor, layer_blocks, 2
        )
        self.stage_3 = self._make_layer(
            WideBasicblock, 64 * widen_factor, layer_blocks, 2
        )
        self.lastact = nn.Sequential(
            nn.BatchNorm2d(64 * widen_factor), nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64 * widen_factor, num_classes)

        self.apply(initialize_resnet)

    def get_message(self):
        return self.message

    def _make_layer(self, block, planes, blocks, stride):

        layers = []
        layers.append(block(self.inplanes, planes, stride, self.dropout))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, self.dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_3x3(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.lastact(x)
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        outs = self.classifier(features)
        return features, outs
