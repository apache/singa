##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##########################################################################
# Efficient Neural Architecture Search via Parameters Sharing, ICML 2018 #
##########################################################################
import torch
import torch.nn as nn
from copy import deepcopy
from ..cell_operations import ResNetBasicblock
from .search_cells import NAS201SearchCell as SearchCell
from .genotypes import Structure
from .search_model_enas_utils import Controller


class TinyNetworkENAS(nn.Module):
    def __init__(
        self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats
    ):
        super(TinyNetworkENAS, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = SearchCell(
                    C_prev,
                    C_curr,
                    1,
                    max_nodes,
                    search_space,
                    affine,
                    track_running_stats,
                )
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert (
                        num_edge == cell.num_edges and edge2index == cell.edge2index
                    ), "invalid {:} vs. {:}.".format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        # to maintain the sampled architecture
        self.sampled_arch = None

    def update_arch(self, _arch):
        if _arch is None:
            self.sampled_arch = None
        elif isinstance(_arch, Structure):
            self.sampled_arch = _arch
        elif isinstance(_arch, (list, tuple)):
            genotypes = []
            for i in range(1, self.max_nodes):
                xlist = []
                for j in range(i):
                    node_str = "{:}<-{:}".format(i, j)
                    op_index = _arch[self.edge2index[node_str]]
                    op_name = self.op_names[op_index]
                    xlist.append((op_name, j))
                genotypes.append(tuple(xlist))
            self.sampled_arch = Structure(genotypes)
        else:
            raise ValueError("invalid type of input architecture : {:}".format(_arch))
        return self.sampled_arch

    def create_controller(self):
        return Controller(len(self.edge2index), len(self.op_names))

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def forward(self, inputs):

        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                feature = cell.forward_dynamic(feature, self.sampled_arch)
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits
