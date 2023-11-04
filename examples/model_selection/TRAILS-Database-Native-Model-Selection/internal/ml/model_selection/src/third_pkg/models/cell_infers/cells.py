#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################

import torch
import torch.nn as nn
from copy import deepcopy

from models.cell_operations import OPS


# Cell for NAS-Bench-201
class InferCell(nn.Module):
    def __init__(
        self, genotype, C_in, C_out, stride, affine=True, track_running_stats=True
    ):
        super(InferCell, self).__init__()

        self.layers = nn.ModuleList()
        self.node_IN = []
        self.node_IX = []
        self.genotype = deepcopy(genotype)
        for i in range(1, len(genotype)):
            node_info = genotype[i - 1]
            cur_index = []
            cur_innod = []
            for (op_name, op_in) in node_info:
                if op_in == 0:
                    layer = OPS[op_name](
                        C_in, C_out, stride, affine, track_running_stats
                    )
                else:
                    layer = OPS[op_name](C_out, C_out, 1, affine, track_running_stats)
                cur_index.append(len(self.layers))
                cur_innod.append(op_in)
                self.layers.append(layer)
            self.node_IX.append(cur_index)
            self.node_IN.append(cur_innod)
        self.nodes = len(genotype)
        self.in_dim = C_in
        self.out_dim = C_out

    def extra_repr(self):
        string = "info :: nodes={nodes}, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__
        )
        laystr = []
        for i, (node_layers, node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            y = [
                "I{:}-L{:}".format(_ii, _il)
                for _il, _ii in zip(node_layers, node_innods)
            ]
            x = "{:}<-({:})".format(i + 1, ",".join(y))
            laystr.append(x)
        return (
            string
            + ", [{:}]".format(" | ".join(laystr))
            + ", {:}".format(self.genotype.tostr())
        )

    def forward(self, inputs):
        nodes = [inputs]
        for i, (node_layers, node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            node_feature = sum(
                self.layers[_il](nodes[_ii])
                for _il, _ii in zip(node_layers, node_innods)
            )
            nodes.append(node_feature)
        return nodes[-1]


# Learning Transferable Architectures for Scalable Image Recognition, CVPR 2018
class NASNetInferCell(nn.Module):
    def __init__(
        self,
        genotype,
        C_prev_prev,
        C_prev,
        C,
        reduction,
        reduction_prev,
        affine,
        track_running_stats,
    ):
        super(NASNetInferCell, self).__init__()
        self.reduction = reduction
        if reduction_prev:
            self.preprocess0 = OPS["skip_connect"](
                C_prev_prev, C, 2, affine, track_running_stats
            )
        else:
            self.preprocess0 = OPS["nor_conv_1x1"](
                C_prev_prev, C, 1, affine, track_running_stats
            )
        self.preprocess1 = OPS["nor_conv_1x1"](
            C_prev, C, 1, affine, track_running_stats
        )

        if not reduction:
            nodes, concats = genotype["normal"], genotype["normal_concat"]
        else:
            nodes, concats = genotype["reduce"], genotype["reduce_concat"]
        self._multiplier = len(concats)
        self._concats = concats
        self._steps = len(nodes)
        self._nodes = nodes
        self.edges = nn.ModuleDict()
        for i, node in enumerate(nodes):
            for in_node in node:
                name, j = in_node[0], in_node[1]
                stride = 2 if reduction and j < 2 else 1
                node_str = "{:}<-{:}".format(i + 2, j)
                self.edges[node_str] = OPS[name](
                    C, C, stride, affine, track_running_stats
                )

    # [TODO] to support drop_prob in this function..
    def forward(self, s0, s1, unused_drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i, node in enumerate(self._nodes):
            clist = []
            for in_node in node:
                name, j = in_node[0], in_node[1]
                node_str = "{:}<-{:}".format(i + 2, j)
                op = self.edges[node_str]
                clist.append(op(states[j]))
            states.append(sum(clist))
        return torch.cat([states[x] for x in self._concats], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(
                5, stride=3, padding=0, count_include_pad=False
            ),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x
