##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import math, torch
from collections import OrderedDict
from bisect import bisect_right
import torch.nn as nn
from ..initialization import initialize_resnet
from ..SharedUtils import additive_func
from .SoftSelect import select2withP, ChannelWiseInter
from .SoftSelect import linear_forward
from .SoftSelect import get_width_choices


def get_depth_choices(nDepth, return_num):
    if nDepth == 2:
        choices = (1, 2)
    elif nDepth == 3:
        choices = (1, 2, 3)
    elif nDepth > 3:
        choices = list(range(1, nDepth + 1, 2))
        if choices[-1] < nDepth:
            choices.append(nDepth)
    else:
        raise ValueError("invalid nDepth : {:}".format(nDepth))
    if return_num:
        return len(choices)
    else:
        return choices


class ConvBNReLU(nn.Module):
    num_conv = 1

    def __init__(
        self, nIn, nOut, kernel, stride, padding, bias, has_avg, has_bn, has_relu
    ):
        super(ConvBNReLU, self).__init__()
        self.InShape = None
        self.OutShape = None
        self.choices = get_width_choices(nOut)
        self.register_buffer("choices_tensor", torch.Tensor(self.choices))

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
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = None
        self.in_dim = nIn
        self.out_dim = nOut

    def get_flops(self, divide=1):
        iC, oC = self.in_dim, self.out_dim
        assert (
            iC <= self.conv.in_channels and oC <= self.conv.out_channels
        ), "{:} vs {:}  |  {:} vs {:}".format(
            iC, self.conv.in_channels, oC, self.conv.out_channels
        )
        assert (
            isinstance(self.InShape, tuple) and len(self.InShape) == 2
        ), "invalid in-shape : {:}".format(self.InShape)
        assert (
            isinstance(self.OutShape, tuple) and len(self.OutShape) == 2
        ), "invalid out-shape : {:}".format(self.OutShape)
        # conv_per_position_flops = self.conv.kernel_size[0] * self.conv.kernel_size[1] * iC * oC / self.conv.groups
        conv_per_position_flops = (
            self.conv.kernel_size[0] * self.conv.kernel_size[1] * 1.0 / self.conv.groups
        )
        all_positions = self.OutShape[0] * self.OutShape[1]
        flops = (conv_per_position_flops * all_positions / divide) * iC * oC
        if self.conv.bias is not None:
            flops += all_positions / divide
        return flops

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
        if self.InShape is None:
            self.InShape = (inputs.size(-2), inputs.size(-1))
            self.OutShape = (out.size(-2), out.size(-1))
        return out


class ResNetBasicblock(nn.Module):
    expansion = 1
    num_conv = 2

    def __init__(self, inplanes, planes, stride):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ConvBNReLU(
            inplanes,
            planes,
            3,
            stride,
            1,
            False,
            has_avg=False,
            has_bn=True,
            has_relu=True,
        )
        self.conv_b = ConvBNReLU(
            planes, planes, 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=False
        )
        if stride == 2:
            self.downsample = ConvBNReLU(
                inplanes,
                planes,
                1,
                1,
                0,
                False,
                has_avg=True,
                has_bn=False,
                has_relu=False,
            )
        elif inplanes != planes:
            self.downsample = ConvBNReLU(
                inplanes,
                planes,
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
        self.out_dim = planes
        self.search_mode = "basic"

    def get_flops(self, divide=1):
        flop_A = self.conv_a.get_flops(divide)
        flop_B = self.conv_b.get_flops(divide)
        if hasattr(self.downsample, "get_flops"):
            flop_C = self.downsample.get_flops(divide)
        else:
            flop_C = 0
        return flop_A + flop_B + flop_C

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, basicblock)
        return nn.functional.relu(out, inplace=True)


class ResNetBottleneck(nn.Module):
    expansion = 4
    num_conv = 3

    def __init__(self, inplanes, planes, stride):
        super(ResNetBottleneck, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_1x1 = ConvBNReLU(
            inplanes, planes, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=True
        )
        self.conv_3x3 = ConvBNReLU(
            planes,
            planes,
            3,
            stride,
            1,
            False,
            has_avg=False,
            has_bn=True,
            has_relu=True,
        )
        self.conv_1x4 = ConvBNReLU(
            planes,
            planes * self.expansion,
            1,
            1,
            0,
            False,
            has_avg=False,
            has_bn=True,
            has_relu=False,
        )
        if stride == 2:
            self.downsample = ConvBNReLU(
                inplanes,
                planes * self.expansion,
                1,
                1,
                0,
                False,
                has_avg=True,
                has_bn=False,
                has_relu=False,
            )
        elif inplanes != planes * self.expansion:
            self.downsample = ConvBNReLU(
                inplanes,
                planes * self.expansion,
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
        self.out_dim = planes * self.expansion
        self.search_mode = "basic"

    def get_range(self):
        return (
            self.conv_1x1.get_range()
            + self.conv_3x3.get_range()
            + self.conv_1x4.get_range()
        )

    def get_flops(self, divide):
        flop_A = self.conv_1x1.get_flops(divide)
        flop_B = self.conv_3x3.get_flops(divide)
        flop_C = self.conv_1x4.get_flops(divide)
        if hasattr(self.downsample, "get_flops"):
            flop_D = self.downsample.get_flops(divide)
        else:
            flop_D = 0
        return flop_A + flop_B + flop_C + flop_D

    def forward(self, inputs):
        bottleneck = self.conv_1x1(inputs)
        bottleneck = self.conv_3x3(bottleneck)
        bottleneck = self.conv_1x4(bottleneck)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, bottleneck)
        return nn.functional.relu(out, inplace=True)


class SearchDepthCifarResNet(nn.Module):
    def __init__(self, block_name, depth, num_classes):
        super(SearchDepthCifarResNet, self).__init__()

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

        self.message = (
            "SearchShapeCifarResNet : Depth : {:} , Layers for each block : {:}".format(
                depth, layer_blocks
            )
        )
        self.num_classes = num_classes
        self.channels = [16]
        self.layers = nn.ModuleList(
            [
                ConvBNReLU(
                    3, 16, 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=True
                )
            ]
        )
        self.InShape = None
        self.depth_info = OrderedDict()
        self.depth_at_i = OrderedDict()
        for stage in range(3):
            cur_block_choices = get_depth_choices(layer_blocks, False)
            assert (
                cur_block_choices[-1] == layer_blocks
            ), "stage={:}, {:} vs {:}".format(stage, cur_block_choices, layer_blocks)
            self.message += (
                "\nstage={:} ::: depth-block-choices={:} for {:} blocks.".format(
                    stage, cur_block_choices, layer_blocks
                )
            )
            block_choices, xstart = [], len(self.layers)
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
                # added for depth
                layer_index = len(self.layers) - 1
                if iL + 1 in cur_block_choices:
                    block_choices.append(layer_index)
                if iL + 1 == layer_blocks:
                    self.depth_info[layer_index] = {
                        "choices": block_choices,
                        "stage": stage,
                        "xstart": xstart,
                    }
        self.depth_info_list = []
        for xend, info in self.depth_info.items():
            self.depth_info_list.append((xend, info))
            xstart, xstage = info["xstart"], info["stage"]
            for ilayer in range(xstart, xend + 1):
                idx = bisect_right(info["choices"], ilayer - 1)
                self.depth_at_i[ilayer] = (xstage, idx)

        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(module.out_dim, num_classes)
        self.InShape = None
        self.tau = -1
        self.search_mode = "basic"
        # assert sum(x.num_conv for x in self.layers) + 1 == depth, 'invalid depth check {:} vs {:}'.format(sum(x.num_conv for x in self.layers)+1, depth)

        self.register_parameter(
            "depth_attentions",
            nn.Parameter(torch.Tensor(3, get_depth_choices(layer_blocks, True))),
        )
        nn.init.normal_(self.depth_attentions, 0, 0.01)
        self.apply(initialize_resnet)

    def arch_parameters(self):
        return [self.depth_attentions]

    def base_parameters(self):
        return (
            list(self.layers.parameters())
            + list(self.avgpool.parameters())
            + list(self.classifier.parameters())
        )

    def get_flop(self, mode, config_dict, extra_info):
        if config_dict is not None:
            config_dict = config_dict.copy()
        # select depth
        if mode == "genotype":
            with torch.no_grad():
                depth_probs = nn.functional.softmax(self.depth_attentions, dim=1)
                choices = torch.argmax(depth_probs, dim=1).cpu().tolist()
        elif mode == "max":
            choices = [depth_probs.size(1) - 1 for _ in range(depth_probs.size(0))]
        elif mode == "random":
            with torch.no_grad():
                depth_probs = nn.functional.softmax(self.depth_attentions, dim=1)
                choices = torch.multinomial(depth_probs, 1, False).cpu().tolist()
        else:
            raise ValueError("invalid mode : {:}".format(mode))
        selected_layers = []
        for choice, xvalue in zip(choices, self.depth_info_list):
            xtemp = xvalue[1]["choices"][choice] - xvalue[1]["xstart"] + 1
            selected_layers.append(xtemp)
        flop = 0
        for i, layer in enumerate(self.layers):
            if i in self.depth_at_i:
                xstagei, xatti = self.depth_at_i[i]
                if xatti <= choices[xstagei]:  # leave this depth
                    flop += layer.get_flops()
                else:
                    flop += 0  # do not use this layer
            else:
                flop += layer.get_flops()
        # the last fc layer
        flop += self.classifier.in_features * self.classifier.out_features
        if config_dict is None:
            return flop / 1e6
        else:
            config_dict["xblocks"] = selected_layers
            config_dict["super_type"] = "infer-depth"
            config_dict["estimated_FLOP"] = flop / 1e6
            return flop / 1e6, config_dict

    def get_arch_info(self):
        string = "for depth, there are {:} attention probabilities.".format(
            len(self.depth_attentions)
        )
        string += "\n{:}".format(self.depth_info)
        discrepancy = []
        with torch.no_grad():
            for i, att in enumerate(self.depth_attentions):
                prob = nn.functional.softmax(att, dim=0)
                prob = prob.cpu()
                selc = prob.argmax().item()
                prob = prob.tolist()
                prob = ["{:.3f}".format(x) for x in prob]
                xstring = "{:03d}/{:03d}-th : {:}".format(
                    i, len(self.depth_attentions), " ".join(prob)
                )
                logt = ["{:.4f}".format(x) for x in att.cpu().tolist()]
                xstring += "  ||  {:17s}".format(" ".join(logt))
                prob = sorted([float(x) for x in prob])
                disc = prob[-1] - prob[-2]
                xstring += "  || discrepancy={:.2f} || select={:}/{:}".format(
                    disc, selc, len(prob)
                )
                discrepancy.append(disc)
                string += "\n{:}".format(xstring)
        return string, discrepancy

    def set_tau(self, tau_max, tau_min, epoch_ratio):
        assert (
            epoch_ratio >= 0 and epoch_ratio <= 1
        ), "invalid epoch-ratio : {:}".format(epoch_ratio)
        tau = tau_min + (tau_max - tau_min) * (1 + math.cos(math.pi * epoch_ratio)) / 2
        self.tau = tau

    def get_message(self):
        return self.message

    def forward(self, inputs):
        if self.search_mode == "basic":
            return self.basic_forward(inputs)
        elif self.search_mode == "search":
            return self.search_forward(inputs)
        else:
            raise ValueError("invalid search_mode = {:}".format(self.search_mode))

    def search_forward(self, inputs):
        flop_depth_probs = nn.functional.softmax(self.depth_attentions, dim=1)
        flop_depth_probs = torch.flip(
            torch.cumsum(torch.flip(flop_depth_probs, [1]), 1), [1]
        )
        selected_depth_probs = select2withP(self.depth_attentions, self.tau, True)

        x, flops = inputs, []
        feature_maps = []
        for i, layer in enumerate(self.layers):
            layer_i = layer(x)
            feature_maps.append(layer_i)
            if i in self.depth_info:  # aggregate the information
                choices = self.depth_info[i]["choices"]
                xstagei = self.depth_info[i]["stage"]
                possible_tensors = []
                for tempi, A in enumerate(choices):
                    xtensor = feature_maps[A]
                    possible_tensors.append(xtensor)
                weighted_sum = sum(
                    xtensor * W
                    for xtensor, W in zip(
                        possible_tensors, selected_depth_probs[xstagei]
                    )
                )
                x = weighted_sum
            else:
                x = layer_i

            if i in self.depth_at_i:
                xstagei, xatti = self.depth_at_i[i]
                # print ('layer-{:03d}, stage={:}, att={:}, prob={:}, flop={:}'.format(i, xstagei, xatti, flop_depth_probs[xstagei, xatti].item(), layer.get_flops(1e6)))
                x_expected_flop = flop_depth_probs[xstagei, xatti] * layer.get_flops(
                    1e6
                )
            else:
                x_expected_flop = layer.get_flops(1e6)
            flops.append(x_expected_flop)
        flops.append(
            (self.classifier.in_features * self.classifier.out_features * 1.0 / 1e6)
        )

        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = linear_forward(features, self.classifier)
        return logits, torch.stack([sum(flops)])

    def basic_forward(self, inputs):
        if self.InShape is None:
            self.InShape = (inputs.size(-2), inputs.size(-1))
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return features, logits
