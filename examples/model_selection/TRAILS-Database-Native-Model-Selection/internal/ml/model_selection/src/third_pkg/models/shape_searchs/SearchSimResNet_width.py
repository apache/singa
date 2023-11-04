##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import math, torch
import torch.nn as nn
from ..initialization import initialize_resnet
from ..SharedUtils import additive_func
from .SoftSelect import select2withP, ChannelWiseInter
from .SoftSelect import linear_forward
from .SoftSelect import get_width_choices as get_choices


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
        self.choices = get_choices(nOut)
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


class SimBlock(nn.Module):
    expansion = 1
    num_conv = 1

    def __init__(self, inplanes, planes, stride):
        super(SimBlock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv = ConvBNReLU(
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
        return self.conv.get_range()

    def get_flops(self, channels):
        assert len(channels) == 2, "invalid channels : {:}".format(channels)
        flop_A = self.conv.get_flops([channels[0], channels[1]])
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
                * self.conv.OutShape[0]
                * self.conv.OutShape[1]
            )
        return flop_A + flop_C

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
        assert (
            indexes.size(0) == 1 and probs.size(0) == 1 and probability.size(0) == 1
        ), "invalid size : {:}, {:}, {:}".format(
            indexes.size(), probs.size(), probability.size()
        )
        out, expected_next_inC, expected_flop = self.conv(
            (inputs, expected_inC, probability[0], indexes[0], probs[0])
        )
        if self.downsample is not None:
            residual, _, expected_flop_c = self.downsample(
                (inputs, expected_inC, probability[-1], indexes[-1], probs[-1])
            )
        else:
            residual, expected_flop_c = inputs, 0
        out = additive_func(residual, out)
        return (
            nn.functional.relu(out, inplace=True),
            expected_next_inC,
            sum([expected_flop, expected_flop_c]),
        )

    def basic_forward(self, inputs):
        basicblock = self.conv(inputs)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, basicblock)
        return nn.functional.relu(out, inplace=True)


class SearchWidthSimResNet(nn.Module):
    def __init__(self, depth, num_classes):
        super(SearchWidthSimResNet, self).__init__()

        assert (
            depth - 2
        ) % 3 == 0, "depth should be one of 5, 8, 11, 14, ... instead of {:}".format(
            depth
        )
        layer_blocks = (depth - 2) // 3
        self.message = (
            "SearchWidthSimResNet : Depth : {:} , Layers for each block : {:}".format(
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
        for stage in range(3):
            for iL in range(layer_blocks):
                iC = self.channels[-1]
                planes = 16 * (2 ** stage)
                stride = 2 if stage > 0 and iL == 0 else 1
                module = SimBlock(iC, planes, stride)
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
            nn.Parameter(torch.Tensor(len(self.Ranges), get_choices(None))),
        )
        nn.init.normal_(self.width_attentions, 0, 0.01)
        self.apply(initialize_resnet)

    def arch_parameters(self):
        return [self.width_attentions]

    def base_parameters(self):
        return (
            list(self.layers.parameters())
            + list(self.avgpool.parameters())
            + list(self.classifier.parameters())
        )

    def get_flop(self, mode, config_dict, extra_info):
        if config_dict is not None:
            config_dict = config_dict.copy()
        # weights = [F.softmax(x, dim=0) for x in self.width_attentions]
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
        flop = 0
        for i, layer in enumerate(self.layers):
            s, e = self.layer2indexRange[i]
            xchl = tuple(channels[s : e + 1])
            flop += layer.get_flops(xchl)
        # the last fc layer
        flop += channels[-1] * self.classifier.out_features
        if config_dict is None:
            return flop / 1e6
        else:
            config_dict["xchannels"] = channels
            config_dict["super_type"] = "infer-width"
            config_dict["estimated_FLOP"] = flop / 1e6
            return flop / 1e6, config_dict

    def get_arch_info(self):
        string = "for width, there are {:} attention probabilities.".format(
            len(self.width_attentions)
        )
        discrepancy = []
        with torch.no_grad():
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
        flop_probs = nn.functional.softmax(self.width_attentions, dim=1)
        selected_widths, selected_probs = select2withP(self.width_attentions, self.tau)
        with torch.no_grad():
            selected_widths = selected_widths.cpu()

        x, last_channel_idx, expected_inC, flops = inputs, 0, 3, []
        for i, layer in enumerate(self.layers):
            selected_w_index = selected_widths[
                last_channel_idx : last_channel_idx + layer.num_conv
            ]
            selected_w_probs = selected_probs[
                last_channel_idx : last_channel_idx + layer.num_conv
            ]
            layer_prob = flop_probs[
                last_channel_idx : last_channel_idx + layer.num_conv
            ]
            x, expected_inC, expected_flop = layer(
                (x, expected_inC, layer_prob, selected_w_index, selected_w_probs)
            )
            last_channel_idx += layer.num_conv
            flops.append(expected_flop)
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
