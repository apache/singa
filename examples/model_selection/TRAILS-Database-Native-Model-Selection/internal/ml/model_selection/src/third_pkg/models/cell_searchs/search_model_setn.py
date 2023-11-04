#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
######################################################################################
# One-Shot Neural Architecture Search via Self-Evaluated Template Network, ICCV 2019 #
######################################################################################
import torch, random
import torch.nn as nn
from copy import deepcopy
from ..cell_operations import ResNetBasicblock
from .search_cells import NAS201SearchCell as SearchCell
from .genotypes import Structure


class TinyNetworkSETN(nn.Module):
    def __init__(
        self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats
    ):
        super(TinyNetworkSETN, self).__init__()
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
        self.arch_parameters = nn.Parameter(
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

    def get_cal_mode(self):
        return self.mode

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(
            self.global_pooling.parameters()
        )
        xlist += list(self.classifier.parameters())
        return xlist

    def get_alphas(self):
        return [self.arch_parameters]

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

    def genotype(self):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                with torch.no_grad():
                    weights = self.arch_parameters[self.edge2index[node_str]]
                    op_name = self.op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

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

    def get_log_prob(self, arch):
        with torch.no_grad():
            logits = nn.functional.log_softmax(self.arch_parameters, dim=-1)
        select_logits = []
        for i, node_info in enumerate(arch.nodes):
            for op, xin in node_info:
                node_str = "{:}<-{:}".format(i + 1, xin)
                op_index = self.op_names.index(op)
                select_logits.append(logits[self.edge2index[node_str], op_index])
        return sum(select_logits).item()

    def return_topK(self, K):
        archs = Structure.gen_all(self.op_names, self.max_nodes, False)
        pairs = [(self.get_log_prob(arch), arch) for arch in archs]
        if K < 0 or K >= len(archs):
            K = len(archs)
        sorted_pairs = sorted(pairs, key=lambda x: -x[0])
        return_pairs = [sorted_pairs[_][1] for _ in range(K)]
        return return_pairs

    def forward(self, inputs):
        alphas = nn.functional.softmax(self.arch_parameters, dim=-1)
        with torch.no_grad():
            alphas_cpu = alphas.detach().cpu()

        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                if self.mode == "urs":
                    feature = cell.forward_urs(feature)
                elif self.mode == "select":
                    feature = cell.forward_select(feature, alphas_cpu)
                elif self.mode == "joint":
                    feature = cell.forward_joint(feature, alphas)
                elif self.mode == "dynamic":
                    feature = cell.forward_dynamic(feature, self.dynamic_cell)
                else:
                    raise ValueError("invalid mode={:}".format(self.mode))
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits
