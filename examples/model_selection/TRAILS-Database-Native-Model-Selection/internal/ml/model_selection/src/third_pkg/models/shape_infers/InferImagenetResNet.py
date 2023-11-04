#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
import torch.nn as nn
import torch.nn.functional as F
from ..initialization import initialize_resnet


class ConvBNReLU(nn.Module):

    num_conv = 1

    def __init__(
        self, nIn, nOut, kernel, stride, padding, bias, has_avg, has_bn, has_relu
    ):
        super(ConvBNReLU, self).__init__()
        if has_avg:
            self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.avg = None
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
            bias=bias,
        )
        if has_bn:
            self.bn = nn.BatchNorm2d(nOut)
        else:
            self.bn = None
        if has_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, inputs):
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        conv = self.conv(out)
        if self.bn:
            out = self.bn(conv)
        else:
            out = conv
        if self.relu:
            out = self.relu(out)
        else:
            out = out

        return out


class ResNetBasicblock(nn.Module):
    num_conv = 2
    expansion = 1

    def __init__(self, iCs, stride):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        assert isinstance(iCs, tuple) or isinstance(
            iCs, list
        ), "invalid type of iCs : {:}".format(iCs)
        assert len(iCs) == 3, "invalid lengths of iCs : {:}".format(iCs)

        self.conv_a = ConvBNReLU(
            iCs[0],
            iCs[1],
            3,
            stride,
            1,
            False,
            has_avg=False,
            has_bn=True,
            has_relu=True,
        )
        self.conv_b = ConvBNReLU(
            iCs[1], iCs[2], 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=False
        )
        residual_in = iCs[0]
        if stride == 2:
            self.downsample = ConvBNReLU(
                iCs[0],
                iCs[2],
                1,
                1,
                0,
                False,
                has_avg=True,
                has_bn=True,
                has_relu=False,
            )
            residual_in = iCs[2]
        elif iCs[0] != iCs[2]:
            self.downsample = ConvBNReLU(
                iCs[0],
                iCs[2],
                1,
                1,
                0,
                False,
                has_avg=False,
                has_bn=True,
                has_relu=False,
            )
        else:
            self.downsample = None
        # self.out_dim  = max(residual_in, iCs[2])
        self.out_dim = iCs[2]

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = residual + basicblock
        return F.relu(out, inplace=True)


class ResNetBottleneck(nn.Module):
    expansion = 4
    num_conv = 3

    def __init__(self, iCs, stride):
        super(ResNetBottleneck, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        assert isinstance(iCs, tuple) or isinstance(
            iCs, list
        ), "invalid type of iCs : {:}".format(iCs)
        assert len(iCs) == 4, "invalid lengths of iCs : {:}".format(iCs)
        self.conv_1x1 = ConvBNReLU(
            iCs[0], iCs[1], 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=True
        )
        self.conv_3x3 = ConvBNReLU(
            iCs[1],
            iCs[2],
            3,
            stride,
            1,
            False,
            has_avg=False,
            has_bn=True,
            has_relu=True,
        )
        self.conv_1x4 = ConvBNReLU(
            iCs[2], iCs[3], 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False
        )
        residual_in = iCs[0]
        if stride == 2:
            self.downsample = ConvBNReLU(
                iCs[0],
                iCs[3],
                1,
                1,
                0,
                False,
                has_avg=True,
                has_bn=True,
                has_relu=False,
            )
            residual_in = iCs[3]
        elif iCs[0] != iCs[3]:
            self.downsample = ConvBNReLU(
                iCs[0],
                iCs[3],
                1,
                1,
                0,
                False,
                has_avg=False,
                has_bn=True,
                has_relu=False,
            )
            residual_in = iCs[3]
        else:
            self.downsample = None
        # self.out_dim = max(residual_in, iCs[3])
        self.out_dim = iCs[3]

    def forward(self, inputs):

        bottleneck = self.conv_1x1(inputs)
        bottleneck = self.conv_3x3(bottleneck)
        bottleneck = self.conv_1x4(bottleneck)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = residual + bottleneck
        return F.relu(out, inplace=True)


class InferImagenetResNet(nn.Module):
    def __init__(
        self,
        block_name,
        layers,
        xblocks,
        xchannels,
        deep_stem,
        num_classes,
        zero_init_residual,
    ):
        super(InferImagenetResNet, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        if block_name == "BasicBlock":
            block = ResNetBasicblock
        elif block_name == "Bottleneck":
            block = ResNetBottleneck
        else:
            raise ValueError("invalid block : {:}".format(block_name))
        assert len(xblocks) == len(
            layers
        ), "invalid layers : {:} vs xblocks : {:}".format(layers, xblocks)

        self.message = "InferImagenetResNet : Depth : {:} -> {:}, Layers for each block : {:}".format(
            sum(layers) * block.num_conv, sum(xblocks) * block.num_conv, xblocks
        )
        self.num_classes = num_classes
        self.xchannels = xchannels
        if not deep_stem:
            self.layers = nn.ModuleList(
                [
                    ConvBNReLU(
                        xchannels[0],
                        xchannels[1],
                        7,
                        2,
                        3,
                        False,
                        has_avg=False,
                        has_bn=True,
                        has_relu=True,
                    )
                ]
            )
            last_channel_idx = 1
        else:
            self.layers = nn.ModuleList(
                [
                    ConvBNReLU(
                        xchannels[0],
                        xchannels[1],
                        3,
                        2,
                        1,
                        False,
                        has_avg=False,
                        has_bn=True,
                        has_relu=True,
                    ),
                    ConvBNReLU(
                        xchannels[1],
                        xchannels[2],
                        3,
                        1,
                        1,
                        False,
                        has_avg=False,
                        has_bn=True,
                        has_relu=True,
                    ),
                ]
            )
            last_channel_idx = 2
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for stage, layer_blocks in enumerate(layers):
            for iL in range(layer_blocks):
                num_conv = block.num_conv
                iCs = self.xchannels[last_channel_idx : last_channel_idx + num_conv + 1]
                stride = 2 if stage > 0 and iL == 0 else 1
                module = block(iCs, stride)
                last_channel_idx += num_conv
                self.xchannels[last_channel_idx] = module.out_dim
                self.layers.append(module)
                self.message += "\nstage={:}, ilayer={:02d}/{:02d}, block={:03d}, iCs={:}, oC={:3d}, stride={:}".format(
                    stage,
                    iL,
                    layer_blocks,
                    len(self.layers) - 1,
                    iCs,
                    module.out_dim,
                    stride,
                )
                if iL + 1 == xblocks[stage]:  # reach the maximum depth
                    out_channel = module.out_dim
                    for iiL in range(iL + 1, layer_blocks):
                        last_channel_idx += num_conv
                    self.xchannels[last_channel_idx] = module.out_dim
                    break
        assert last_channel_idx + 1 == len(self.xchannels), "{:} vs {:}".format(
            last_channel_idx, len(self.xchannels)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.xchannels[-1], num_classes)

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
