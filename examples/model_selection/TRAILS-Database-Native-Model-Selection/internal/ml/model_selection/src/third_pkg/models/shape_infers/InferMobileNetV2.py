#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
# MobileNetV2: Inverted Residuals and Linear Bottlenecks, CVPR 2018
#####################################################
from torch import nn

from ..initialization import initialize_resnet
from ..SharedUtils import parse_channel_info


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride,
        groups,
        has_bn=True,
        has_relu=True,
    ):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        if has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        else:
            self.bn = None
        if has_relu:
            self.relu = nn.ReLU6(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        out = self.conv(x)
        if self.bn:
            out = self.bn(out)
        if self.relu:
            out = self.relu(out)
        return out


class InvertedResidual(nn.Module):
    def __init__(self, channels, stride, expand_ratio, additive):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], "invalid stride : {:}".format(stride)
        assert len(channels) in [2, 3], "invalid channels : {:}".format(channels)

        if len(channels) == 2:
            layers = []
        else:
            layers = [ConvBNReLU(channels[0], channels[1], 1, 1, 1)]
        layers.extend(
            [
                # dw
                ConvBNReLU(channels[-2], channels[-2], 3, stride, channels[-2]),
                # pw-linear
                ConvBNReLU(channels[-2], channels[-1], 1, 1, 1, True, False),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.additive = additive
        if self.additive and channels[0] != channels[-1]:
            self.shortcut = ConvBNReLU(channels[0], channels[-1], 1, 1, 1, True, False)
        else:
            self.shortcut = None
        self.out_dim = channels[-1]

    def forward(self, x):
        out = self.conv(x)
        # if self.additive: return additive_func(out, x)
        if self.shortcut:
            return out + self.shortcut(x)
        else:
            return out


class InferMobileNetV2(nn.Module):
    def __init__(self, num_classes, xchannels, xblocks, dropout):
        super(InferMobileNetV2, self).__init__()
        block = InvertedResidual
        inverted_residual_setting = [
            # t, c,  n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        assert len(inverted_residual_setting) == len(
            xblocks
        ), "invalid number of layers : {:} vs {:}".format(
            len(inverted_residual_setting), len(xblocks)
        )
        for block_num, ir_setting in zip(xblocks, inverted_residual_setting):
            assert block_num <= ir_setting[2], "{:} vs {:}".format(
                block_num, ir_setting
            )
        xchannels = parse_channel_info(xchannels)
        # for i, chs in enumerate(xchannels):
        #  if i > 0: assert chs[0] == xchannels[i-1][-1], 'Layer[{:}] is invalid {:} vs {:}'.format(i, xchannels[i-1], chs)
        self.xchannels = xchannels
        self.message = "InferMobileNetV2 : xblocks={:}".format(xblocks)
        # building first layer
        features = [ConvBNReLU(xchannels[0][0], xchannels[0][1], 3, 2, 1)]
        last_channel_idx = 1

        # building inverted residual blocks
        for stage, (t, c, n, s) in enumerate(inverted_residual_setting):
            for i in range(n):
                stride = s if i == 0 else 1
                additv = True if i > 0 else False
                module = block(self.xchannels[last_channel_idx], stride, t, additv)
                features.append(module)
                self.message += "\nstage={:}, ilayer={:02d}/{:02d}, block={:03d}, Cs={:}, stride={:}, expand={:}, original-C={:}".format(
                    stage,
                    i,
                    n,
                    len(features),
                    self.xchannels[last_channel_idx],
                    stride,
                    t,
                    c,
                )
                last_channel_idx += 1
                if i + 1 == xblocks[stage]:
                    out_channel = module.out_dim
                    for iiL in range(i + 1, n):
                        last_channel_idx += 1
                    self.xchannels[last_channel_idx][0] = module.out_dim
                    break
        # building last several layers
        features.append(
            ConvBNReLU(
                self.xchannels[last_channel_idx][0],
                self.xchannels[last_channel_idx][1],
                1,
                1,
                1,
            )
        )
        assert last_channel_idx + 2 == len(self.xchannels), "{:} vs {:}".format(
            last_channel_idx, len(self.xchannels)
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.xchannels[last_channel_idx][1], num_classes),
        )

        # weight initialization
        self.apply(initialize_resnet)

    def get_message(self):
        return self.message

    def forward(self, inputs):
        features = self.features(inputs)
        vectors = features.mean([2, 3])
        predicts = self.classifier(vectors)
        return features, predicts
