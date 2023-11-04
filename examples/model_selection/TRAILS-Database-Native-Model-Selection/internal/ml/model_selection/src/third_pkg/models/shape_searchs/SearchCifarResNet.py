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


def conv_forward(inputs, conv, choices):
    iC = conv.in_channels
    fill_size = list(inputs.size())
    fill_size[1] = iC - fill_size[1]
    filled = torch.zeros(fill_size, device=inputs.device)
    xinputs = torch.cat((inputs, filled), dim=1)
    outputs = conv(xinputs)
    selecteds = [outputs[:, :oC] for oC in choices]
    return selecteds


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
        # if has_bn  : self.bn  = nn.BatchNorm2d(nOut)
        # else       : self.bn  = None
        self.has_bn = has_bn
        self.BNs = nn.ModuleList()
        for i, _out in enumerate(self.choices):
            self.BNs.append(nn.BatchNorm2d(_out))
        if has_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None
        self.in_dim = nIn
        self.out_dim = nOut
        self.search_mode = "basic"

    def get_flops(self, channels, check_range=True, divide=1):
        iC, oC = channels
        if check_range:
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

    def get_range(self):
        return [self.choices]

    def forward(self, inputs):
        if self.search_mode == "basic":
            return self.basic_forward(inputs)
        elif self.search_mode == "search":
            return self.search_forward(inputs)
        else:
            raise ValueError("invalid search_mode = {:}".format(self.search_mode))

    def search_forward(self, tuple_inputs):
        assert (
            isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 5
        ), "invalid type input : {:}".format(type(tuple_inputs))
        inputs, expected_inC, probability, index, prob = tuple_inputs
        index, prob = torch.squeeze(index).tolist(), torch.squeeze(prob)
        probability = torch.squeeze(probability)
        assert len(index) == 2, "invalid length : {:}".format(index)
        # compute expected flop
        # coordinates   = torch.arange(self.x_range[0], self.x_range[1]+1).type_as(probability)
        expected_outC = (self.choices_tensor * probability).sum()
        expected_flop = self.get_flops([expected_inC, expected_outC], False, 1e6)
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        # convolutional layer
        out_convs = conv_forward(out, self.conv, [self.choices[i] for i in index])
        out_bns = [self.BNs[idx](out_conv) for idx, out_conv in zip(index, out_convs)]
        # merge
        out_channel = max([x.size(1) for x in out_bns])
        outA = ChannelWiseInter(out_bns[0], out_channel)
        outB = ChannelWiseInter(out_bns[1], out_channel)
        out = outA * prob[0] + outB * prob[1]
        # out = additive_func(out_bns[0]*prob[0], out_bns[1]*prob[1])

        if self.relu:
            out = self.relu(out)
        else:
            out = out
        return out, expected_outC, expected_flop

    def basic_forward(self, inputs):
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        conv = self.conv(out)
        if self.has_bn:
            out = self.BNs[-1](conv)
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

    def get_range(self):
        return self.conv_a.get_range() + self.conv_b.get_range()

    def get_flops(self, channels):
        assert len(channels) == 3, "invalid channels : {:}".format(channels)
        flop_A = self.conv_a.get_flops([channels[0], channels[1]])
        flop_B = self.conv_b.get_flops([channels[1], channels[2]])
        if hasattr(self.downsample, "get_flops"):
            flop_C = self.downsample.get_flops([channels[0], channels[-1]])
        else:
            flop_C = 0
        if (
            channels[0] != channels[-1] and self.downsample is None
        ):  # this short-cut will be added during the infer-train
            flop_C = (
                channels[0]
                * channels[-1]
                * self.conv_b.OutShape[0]
                * self.conv_b.OutShape[1]
            )
        return flop_A + flop_B + flop_C

    def forward(self, inputs):
        if self.search_mode == "basic":
            return self.basic_forward(inputs)
        elif self.search_mode == "search":
            return self.search_forward(inputs)
        else:
            raise ValueError("invalid search_mode = {:}".format(self.search_mode))

    def search_forward(self, tuple_inputs):
        assert (
            isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 5
        ), "invalid type input : {:}".format(type(tuple_inputs))
        inputs, expected_inC, probability, indexes, probs = tuple_inputs
        assert indexes.size(0) == 2 and probs.size(0) == 2 and probability.size(0) == 2
        out_a, expected_inC_a, expected_flop_a = self.conv_a(
            (inputs, expected_inC, probability[0], indexes[0], probs[0])
        )
        out_b, expected_inC_b, expected_flop_b = self.conv_b(
            (out_a, expected_inC_a, probability[1], indexes[1], probs[1])
        )
        if self.downsample is not None:
            residual, _, expected_flop_c = self.downsample(
                (inputs, expected_inC, probability[1], indexes[1], probs[1])
            )
        else:
            residual, expected_flop_c = inputs, 0
        out = additive_func(residual, out_b)
        return (
            nn.functional.relu(out, inplace=True),
            expected_inC_b,
            sum([expected_flop_a, expected_flop_b, expected_flop_c]),
        )

    def basic_forward(self, inputs):
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

    def get_flops(self, channels):
        assert len(channels) == 4, "invalid channels : {:}".format(channels)
        flop_A = self.conv_1x1.get_flops([channels[0], channels[1]])
        flop_B = self.conv_3x3.get_flops([channels[1], channels[2]])
        flop_C = self.conv_1x4.get_flops([channels[2], channels[3]])
        if hasattr(self.downsample, "get_flops"):
            flop_D = self.downsample.get_flops([channels[0], channels[-1]])
        else:
            flop_D = 0
        if (
            channels[0] != channels[-1] and self.downsample is None
        ):  # this short-cut will be added during the infer-train
            flop_D = (
                channels[0]
                * channels[-1]
                * self.conv_1x4.OutShape[0]
                * self.conv_1x4.OutShape[1]
            )
        return flop_A + flop_B + flop_C + flop_D

    def forward(self, inputs):
        if self.search_mode == "basic":
            return self.basic_forward(inputs)
        elif self.search_mode == "search":
            return self.search_forward(inputs)
        else:
            raise ValueError("invalid search_mode = {:}".format(self.search_mode))

    def basic_forward(self, inputs):
        bottleneck = self.conv_1x1(inputs)
        bottleneck = self.conv_3x3(bottleneck)
        bottleneck = self.conv_1x4(bottleneck)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, bottleneck)
        return nn.functional.relu(out, inplace=True)

    def search_forward(self, tuple_inputs):
        assert (
            isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 5
        ), "invalid type input : {:}".format(type(tuple_inputs))
        inputs, expected_inC, probability, indexes, probs = tuple_inputs
        assert indexes.size(0) == 3 and probs.size(0) == 3 and probability.size(0) == 3
        out_1x1, expected_inC_1x1, expected_flop_1x1 = self.conv_1x1(
            (inputs, expected_inC, probability[0], indexes[0], probs[0])
        )
        out_3x3, expected_inC_3x3, expected_flop_3x3 = self.conv_3x3(
            (out_1x1, expected_inC_1x1, probability[1], indexes[1], probs[1])
        )
        out_1x4, expected_inC_1x4, expected_flop_1x4 = self.conv_1x4(
            (out_3x3, expected_inC_3x3, probability[2], indexes[2], probs[2])
        )
        if self.downsample is not None:
            residual, _, expected_flop_c = self.downsample(
                (inputs, expected_inC, probability[2], indexes[2], probs[2])
            )
        else:
            residual, expected_flop_c = inputs, 0
        out = additive_func(residual, out_1x4)
        return (
            nn.functional.relu(out, inplace=True),
            expected_inC_1x4,
            sum(
                [
                    expected_flop_1x1,
                    expected_flop_3x3,
                    expected_flop_1x4,
                    expected_flop_c,
                ]
            ),
        )


class SearchShapeCifarResNet(nn.Module):
    def __init__(self, block_name, depth, num_classes):
        super(SearchShapeCifarResNet, self).__init__()

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

        # parameters for width
        self.Ranges = []
        self.layer2indexRange = []
        for i, layer in enumerate(self.layers):
            start_index = len(self.Ranges)
            self.Ranges += layer.get_range()
            self.layer2indexRange.append((start_index, len(self.Ranges)))
        assert len(self.Ranges) + 1 == depth, "invalid depth check {:} vs {:}".format(
            len(self.Ranges) + 1, depth
        )

        self.register_parameter(
            "width_attentions",
            nn.Parameter(torch.Tensor(len(self.Ranges), get_width_choices(None))),
        )
        self.register_parameter(
            "depth_attentions",
            nn.Parameter(torch.Tensor(3, get_depth_choices(layer_blocks, True))),
        )
        nn.init.normal_(self.width_attentions, 0, 0.01)
        nn.init.normal_(self.depth_attentions, 0, 0.01)
        self.apply(initialize_resnet)

    def arch_parameters(self, LR=None):
        if LR is None:
            return [self.width_attentions, self.depth_attentions]
        else:
            return [
                {"params": self.width_attentions, "lr": LR},
                {"params": self.depth_attentions, "lr": LR},
            ]

    def base_parameters(self):
        return (
            list(self.layers.parameters())
            + list(self.avgpool.parameters())
            + list(self.classifier.parameters())
        )

    def get_flop(self, mode, config_dict, extra_info):
        if config_dict is not None:
            config_dict = config_dict.copy()
        # select channels
        channels = [3]
        for i, weight in enumerate(self.width_attentions):
            if mode == "genotype":
                with torch.no_grad():
                    probe = nn.functional.softmax(weight, dim=0)
                    C = self.Ranges[i][torch.argmax(probe).item()]
            elif mode == "max":
                C = self.Ranges[i][-1]
            elif mode == "fix":
                C = int(math.sqrt(extra_info) * self.Ranges[i][-1])
            elif mode == "random":
                assert isinstance(extra_info, float), "invalid extra_info : {:}".format(
                    extra_info
                )
                with torch.no_grad():
                    prob = nn.functional.softmax(weight, dim=0)
                    approximate_C = int(math.sqrt(extra_info) * self.Ranges[i][-1])
                    for j in range(prob.size(0)):
                        prob[j] = 1 / (
                            abs(j - (approximate_C - self.Ranges[i][j])) + 0.2
                        )
                    C = self.Ranges[i][torch.multinomial(prob, 1, False).item()]
            else:
                raise ValueError("invalid mode : {:}".format(mode))
            channels.append(C)
        # select depth
        if mode == "genotype":
            with torch.no_grad():
                depth_probs = nn.functional.softmax(self.depth_attentions, dim=1)
                choices = torch.argmax(depth_probs, dim=1).cpu().tolist()
        elif mode == "max" or mode == "fix":
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
            s, e = self.layer2indexRange[i]
            xchl = tuple(channels[s : e + 1])
            if i in self.depth_at_i:
                xstagei, xatti = self.depth_at_i[i]
                if xatti <= choices[xstagei]:  # leave this depth
                    flop += layer.get_flops(xchl)
                else:
                    flop += 0  # do not use this layer
            else:
                flop += layer.get_flops(xchl)
        # the last fc layer
        flop += channels[-1] * self.classifier.out_features
        if config_dict is None:
            return flop / 1e6
        else:
            config_dict["xchannels"] = channels
            config_dict["xblocks"] = selected_layers
            config_dict["super_type"] = "infer-shape"
            config_dict["estimated_FLOP"] = flop / 1e6
            return flop / 1e6, config_dict

    def get_arch_info(self):
        string = (
            "for depth and width, there are {:} + {:} attention probabilities.".format(
                len(self.depth_attentions), len(self.width_attentions)
            )
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
            string += "\n-----------------------------------------------"
            for i, att in enumerate(self.width_attentions):
                prob = nn.functional.softmax(att, dim=0)
                prob = prob.cpu()
                selc = prob.argmax().item()
                prob = prob.tolist()
                prob = ["{:.3f}".format(x) for x in prob]
                xstring = "{:03d}/{:03d}-th : {:}".format(
                    i, len(self.width_attentions), " ".join(prob)
                )
                logt = ["{:.3f}".format(x) for x in att.cpu().tolist()]
                xstring += "  ||  {:52s}".format(" ".join(logt))
                prob = sorted([float(x) for x in prob])
                disc = prob[-1] - prob[-2]
                xstring += "  || dis={:.2f} || select={:}/{:}".format(
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
        flop_width_probs = nn.functional.softmax(self.width_attentions, dim=1)
        flop_depth_probs = nn.functional.softmax(self.depth_attentions, dim=1)
        flop_depth_probs = torch.flip(
            torch.cumsum(torch.flip(flop_depth_probs, [1]), 1), [1]
        )
        selected_widths, selected_width_probs = select2withP(
            self.width_attentions, self.tau
        )
        selected_depth_probs = select2withP(self.depth_attentions, self.tau, True)
        with torch.no_grad():
            selected_widths = selected_widths.cpu()

        x, last_channel_idx, expected_inC, flops = inputs, 0, 3, []
        feature_maps = []
        for i, layer in enumerate(self.layers):
            selected_w_index = selected_widths[
                last_channel_idx : last_channel_idx + layer.num_conv
            ]
            selected_w_probs = selected_width_probs[
                last_channel_idx : last_channel_idx + layer.num_conv
            ]
            layer_prob = flop_width_probs[
                last_channel_idx : last_channel_idx + layer.num_conv
            ]
            x, expected_inC, expected_flop = layer(
                (x, expected_inC, layer_prob, selected_w_index, selected_w_probs)
            )
            feature_maps.append(x)
            last_channel_idx += layer.num_conv
            if i in self.depth_info:  # aggregate the information
                choices = self.depth_info[i]["choices"]
                xstagei = self.depth_info[i]["stage"]
                # print ('iL={:}, choices={:}, stage={:}, probs={:}'.format(i, choices, xstagei, selected_depth_probs[xstagei].cpu().tolist()))
                # for A, W in zip(choices, selected_depth_probs[xstagei]):
                #  print('Size = {:}, W = {:}'.format(feature_maps[A].size(), W))
                possible_tensors = []
                max_C = max(feature_maps[A].size(1) for A in choices)
                for tempi, A in enumerate(choices):
                    xtensor = ChannelWiseInter(feature_maps[A], max_C)
                    # drop_ratio = 1-(tempi+1.0)/len(choices)
                    # xtensor = drop_path(xtensor, drop_ratio)
                    possible_tensors.append(xtensor)
                weighted_sum = sum(
                    xtensor * W
                    for xtensor, W in zip(
                        possible_tensors, selected_depth_probs[xstagei]
                    )
                )
                x = weighted_sum

            if i in self.depth_at_i:
                xstagei, xatti = self.depth_at_i[i]
                x_expected_flop = flop_depth_probs[xstagei, xatti] * expected_flop
            else:
                x_expected_flop = expected_flop
            flops.append(x_expected_flop)
        flops.append(expected_inC * (self.classifier.out_features * 1.0 / 1e6))
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
