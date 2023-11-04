##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import math, random, torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from ..cell_operations import OPS


# This module is used for NAS-Bench-201, represents a small search space with a complete DAG
class NAS201SearchCell(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        stride,
        max_nodes,
        op_names,
        affine=False,
        track_running_stats=True,
    ):
        super(NAS201SearchCell, self).__init__()

        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                if j == 0:
                    xlists = [
                        OPS[op_name](C_in, C_out, stride, affine, track_running_stats)
                        for op_name in op_names
                    ]
                else:
                    xlists = [
                        OPS[op_name](C_in, C_out, 1, affine, track_running_stats)
                        for op_name in op_names
                    ]
                self.edges[node_str] = nn.ModuleList(xlists)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def extra_repr(self):
        string = "info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__
        )
        return string

    def forward(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(
                    sum(
                        layer(nodes[j]) * w
                        for layer, w in zip(self.edges[node_str], weights)
                    )
                )
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # GDAS
    def forward_gdas(self, inputs, hardwts, index):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = hardwts[self.edge2index[node_str]]
                argmaxs = index[self.edge2index[node_str]].item()
                weigsum = sum(
                    weights[_ie] * edge(nodes[j]) if _ie == argmaxs else weights[_ie]
                    for _ie, edge in enumerate(self.edges[node_str])
                )
                inter_nodes.append(weigsum)
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # GDAS Variant: https://github.com/D-X-Y/AutoDL-Projects/issues/119
    def forward_gdas_v1(self, inputs, hardwts, index):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = hardwts[self.edge2index[node_str]]
                argmaxs = index[self.edge2index[node_str]].item()
                weigsum = weights[argmaxs] * self.edges[node_str](nodes[j])
                inter_nodes.append(weigsum)
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # joint
    def forward_joint(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = weightss[self.edge2index[node_str]]
                # aggregation = sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) / weights.numel()
                aggregation = sum(
                    layer(nodes[j]) * w
                    for layer, w in zip(self.edges[node_str], weights)
                )
                inter_nodes.append(aggregation)
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # uniform random sampling per iteration, SETN
    def forward_urs(self, inputs):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            while True:  # to avoid select zero for all ops
                sops, has_non_zero = [], False
                for j in range(i):
                    node_str = "{:}<-{:}".format(i, j)
                    candidates = self.edges[node_str]
                    select_op = random.choice(candidates)
                    sops.append(select_op)
                    if not hasattr(select_op, "is_zero") or select_op.is_zero is False:
                        has_non_zero = True
                if has_non_zero:
                    break
            inter_nodes = []
            for j, select_op in enumerate(sops):
                inter_nodes.append(select_op(nodes[j]))
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # select the argmax
    def forward_select(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(
                    self.edges[node_str][weights.argmax().item()](nodes[j])
                )
                # inter_nodes.append( sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) )
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # forward with a specific structure
    def forward_dynamic(self, inputs, structure):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            cur_op_node = structure.nodes[i - 1]
            inter_nodes = []
            for op_name, j in cur_op_node:
                node_str = "{:}<-{:}".format(i, j)
                op_index = self.op_names.index(op_name)
                inter_nodes.append(self.edges[node_str][op_index](nodes[j]))
            nodes.append(sum(inter_nodes))
        return nodes[-1]


# Learning Transferable Architectures for Scalable Image Recognition, CVPR 2018


class MixedOp(nn.Module):
    def __init__(self, space, C, stride, affine, track_running_stats):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in space:
            op = OPS[primitive](C, C, stride, affine, track_running_stats)
            self._ops.append(op)

    def forward_gdas(self, x, weights, index):
        return self._ops[index](x) * weights[index]

    def forward_darts(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class NASNetSearchCell(nn.Module):
    def __init__(
        self,
        space,
        steps,
        multiplier,
        C_prev_prev,
        C_prev,
        C,
        reduction,
        reduction_prev,
        affine,
        track_running_stats,
    ):
        super(NASNetSearchCell, self).__init__()
        self.reduction = reduction
        self.op_names = deepcopy(space)
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
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self.edges = nn.ModuleDict()
        for i in range(self._steps):
            for j in range(2 + i):
                node_str = "{:}<-{:}".format(
                    i, j
                )  # indicate the edge from node-(j) to node-(i+2)
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(space, C, stride, affine, track_running_stats)
                self.edges[node_str] = op
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    @property
    def multiplier(self):
        return self._multiplier

    def forward_gdas(self, s0, s1, weightss, indexs):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            clist = []
            for j, h in enumerate(states):
                node_str = "{:}<-{:}".format(i, j)
                op = self.edges[node_str]
                weights = weightss[self.edge2index[node_str]]
                index = indexs[self.edge2index[node_str]].item()
                clist.append(op.forward_gdas(h, weights, index))
            states.append(sum(clist))

        return torch.cat(states[-self._multiplier :], dim=1)

    def forward_darts(self, s0, s1, weightss):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            clist = []
            for j, h in enumerate(states):
                node_str = "{:}<-{:}".format(i, j)
                op = self.edges[node_str]
                weights = weightss[self.edge2index[node_str]]
                clist.append(op.forward_darts(h, weights))
            states.append(sum(clist))

        return torch.cat(states[-self._multiplier :], dim=1)
