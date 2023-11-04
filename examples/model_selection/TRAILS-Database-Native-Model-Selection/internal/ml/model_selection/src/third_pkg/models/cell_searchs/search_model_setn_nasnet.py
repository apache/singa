#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
######################################################################################
# One-Shot Neural Architecture Search via Self-Evaluated Template Network, ICCV 2019 #
######################################################################################
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Text, Dict
from .search_cells import NASNetSearchCell as SearchCell


# The macro structure is based on NASNet
class NASNetworkSETN(nn.Module):
    def __init__(
        self,
        C: int,
        N: int,
        steps: int,
        multiplier: int,
        stem_multiplier: int,
        num_classes: int,
        search_space: List[Text],
        affine: bool,
        track_running_stats: bool,
    ):
        super(NASNetworkSETN, self).__init__()
        self._C = C
        self._layerN = N
        self._steps = steps
        self._multiplier = multiplier
        self.stem = nn.Sequential(
            nn.Conv2d(3, C * stem_multiplier, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C * stem_multiplier),
        )

        # config for each layer
        layer_channels = (
            [C] * N + [C * 2] + [C * 2] * (N - 1) + [C * 4] + [C * 4] * (N - 1)
        )
        layer_reductions = (
            [False] * N + [True] + [False] * (N - 1) + [True] + [False] * (N - 1)
        )

        num_edge, edge2index = None, None
        C_prev_prev, C_prev, C_curr, reduction_prev = (
            C * stem_multiplier,
            C * stem_multiplier,
            C,
            False,
        )

        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            cell = SearchCell(
                search_space,
                steps,
                multiplier,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
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
            C_prev_prev, C_prev, reduction_prev = C_prev, multiplier * C_curr, reduction
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_normal_parameters = nn.Parameter(
            1e-3 * torch.randn(num_edge, len(search_space))
        )
        self.arch_reduce_parameters = nn.Parameter(
            1e-3 * torch.randn(num_edge, len(search_space))
        )
        self.mode = "urs"
        self.dynamic_cell = None

    def set_cal_mode(self, mode, dynamic_cell=None):
        assert mode in ["urs", "joint", "select", "dynamic"]
        self.mode = mode
        if mode == "dynamic":
            self.dynamic_cell = deepcopy(dynamic_cell)
        else:
            self.dynamic_cell = None

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(
            self.global_pooling.parameters()
        )
        xlist += list(self.classifier.parameters())
        return xlist

    def get_alphas(self):
        return [self.arch_normal_parameters, self.arch_reduce_parameters]

    def show_alphas(self):
        with torch.no_grad():
            A = "arch-normal-parameters :\n{:}".format(
                nn.functional.softmax(self.arch_normal_parameters, dim=-1).cpu()
            )
            B = "arch-reduce-parameters :\n{:}".format(
                nn.functional.softmax(self.arch_reduce_parameters, dim=-1).cpu()
            )
        return "{:}\n{:}".format(A, B)

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_C}, N={_layerN}, steps={_steps}, multiplier={_multiplier}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def dync_genotype(self, use_random=False):
        genotypes = []
        with torch.no_grad():
            alphas_cpu = nn.functional.softmax(self.arch_parameters, dim=-1)
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                if use_random:
                    op_name = random.choice(self.op_names)
                else:
                    weights = alphas_cpu[self.edge2index[node_str]]
                    op_index = torch.multinomial(weights, 1).item()
                    op_name = self.op_names[op_index]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def genotype(self):
        def _parse(weights):
            gene = []
            for i in range(self._steps):
                edges = []
                for j in range(2 + i):
                    node_str = "{:}<-{:}".format(i, j)
                    ws = weights[self.edge2index[node_str]]
                    for k, op_name in enumerate(self.op_names):
                        if op_name == "none":
                            continue
                        edges.append((op_name, j, ws[k]))
                edges = sorted(edges, key=lambda x: -x[-1])
                selected_edges = edges[:2]
                gene.append(tuple(selected_edges))
            return gene

        with torch.no_grad():
            gene_normal = _parse(
                torch.softmax(self.arch_normal_parameters, dim=-1).cpu().numpy()
            )
            gene_reduce = _parse(
                torch.softmax(self.arch_reduce_parameters, dim=-1).cpu().numpy()
            )
        return {
            "normal": gene_normal,
            "normal_concat": list(
                range(2 + self._steps - self._multiplier, self._steps + 2)
            ),
            "reduce": gene_reduce,
            "reduce_concat": list(
                range(2 + self._steps - self._multiplier, self._steps + 2)
            ),
        }

    def forward(self, inputs):
        normal_hardwts = nn.functional.softmax(self.arch_normal_parameters, dim=-1)
        reduce_hardwts = nn.functional.softmax(self.arch_reduce_parameters, dim=-1)

        s0 = s1 = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            # [TODO]
            raise NotImplementedError
            if cell.reduction:
                hardwts, index = reduce_hardwts, reduce_index
            else:
                hardwts, index = normal_hardwts, normal_index
            s0, s1 = s1, cell.forward_gdas(s0, s1, hardwts, index)
        out = self.lastact(s1)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits
