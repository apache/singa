import torch
import torch.nn as nn
import torch.nn.functional as F
from .initialization import initialize_resnet
from .SharedUtils import additive_func


class Downsample(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(Downsample, self).__init__()
        assert stride == 2 and nOut == 2 * nIn, "stride:{} IO:{},{}".format(
            stride, nIn, nOut
        )
        self.in_dim = nIn
        self.out_dim = nOut
        self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.avg(x)
        out = self.conv(x)
        return out


class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, kernel, stride, padding, bias, relu):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            nIn, nOut, kernel_size=kernel, stride=stride, padding=padding, bias=bias
        )
        self.bn = nn.BatchNorm2d(nOut)
        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None
        self.out_dim = nOut
        self.num_conv = 1

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        if self.relu:
            return self.relu(bn)
        else:
            return bn


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ConvBNReLU(inplanes, planes, 3, stride, 1, False, True)
        self.conv_b = ConvBNReLU(planes, planes, 3, 1, 1, False, False)
        if stride == 2:
            self.downsample = Downsample(inplanes, planes, stride)
        elif inplanes != planes:
            self.downsample = ConvBNReLU(inplanes, planes, 1, 1, 0, False, False)
        else:
            self.downsample = None
        self.out_dim = planes
        self.num_conv = 2

    def forward(self, inputs):

        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, basicblock)
        return F.relu(out, inplace=True)


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride):
        super(ResNetBottleneck, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_1x1 = ConvBNReLU(inplanes, planes, 1, 1, 0, False, True)
        self.conv_3x3 = ConvBNReLU(planes, planes, 3, stride, 1, False, True)
        self.conv_1x4 = ConvBNReLU(
            planes, planes * self.expansion, 1, 1, 0, False, False
        )
        if stride == 2:
            self.downsample = Downsample(inplanes, planes * self.expansion, stride)
        elif inplanes != planes * self.expansion:
            self.downsample = ConvBNReLU(
                inplanes, planes * self.expansion, 1, 1, 0, False, False
            )
        else:
            self.downsample = None
        self.out_dim = planes * self.expansion
        self.num_conv = 3

    def forward(self, inputs):

        bottleneck = self.conv_1x1(inputs)
        bottleneck = self.conv_3x3(bottleneck)
        bottleneck = self.conv_1x4(bottleneck)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, bottleneck)
        return F.relu(out, inplace=True)


class CifarResNet(nn.Module):
    def __init__(self, block_name, depth, num_classes, zero_init_residual):
        super(CifarResNet, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        if block_name == "ResNetBasicblock":
            block = ResNetBasicblock
            assert (depth - 2) % 6 == 0, "depth should be one of 20, 32, 44, 56, 110"
            layer_blocks = (depth - 2) // 6
        elif block_name == "ResNetBottleneck":
            block = ResNetBottleneck
            assert (depth - 2) % 9 == 0, "depth should be one of 164"
            layer_blocks = (depth - 2) // 9
        else:
            raise ValueError("invalid block : {:}".format(block_name))

        self.message = "CifarResNet : Block : {:}, Depth : {:}, Layers for each block : {:}".format(
            block_name, depth, layer_blocks
        )
        self.num_classes = num_classes
        self.channels = [16]
        self.layers = nn.ModuleList([ConvBNReLU(3, 16, 3, 1, 1, False, True)])
        for stage in range(3):
            for iL in range(layer_blocks):
                iC = self.channels[-1]
                planes = 16 * (2 ** stage)
                stride = 2 if stage > 0 and iL == 0 else 1
                module = block(iC, planes, stride)
                self.channels.append(module.out_dim)
                self.layers.append(module)
                self.message += "\nstage={:}, ilayer={:02d}/{:02d}, block={:03d}, iC={:3d}, oC={:3d}, stride={:}".format(
                    stage,
                    iL,
                    layer_blocks,
                    len(self.layers) - 1,
                    iC,
                    module.out_dim,
                    stride,
                )

        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(module.out_dim, num_classes)
        assert (
            sum(x.num_conv for x in self.layers) + 1 == depth
        ), "invalid depth check {:} vs {:}".format(
            sum(x.num_conv for x in self.layers) + 1, depth
        )

        self.apply(initialize_resnet)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetBasicblock):
                    nn.init.constant_(m.conv_b.bn.weight, 0)
                elif isinstance(m, ResNetBottleneck):
                    nn.init.constant_(m.conv_1x4.bn.weight, 0)

    def get_message(self):
        return self.message

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return features, logits
