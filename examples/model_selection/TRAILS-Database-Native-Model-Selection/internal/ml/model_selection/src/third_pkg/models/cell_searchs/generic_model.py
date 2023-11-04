#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.07 #
#####################################################
import torch, random
import torch.nn as nn
from copy import deepcopy
from typing import Text
from torch.distributions.categorical import Categorical

from ..cell_operations import ResNetBasicblock, drop_path
from .search_cells import NAS201SearchCell as SearchCell
from .genotypes import Structure


class Controller(nn.Module):
    # we refer to https://github.com/TDeVries/enas_pytorch/blob/master/models/controller.py
    def __init__(
        self,
        edge2index,
        op_names,
        max_nodes,
        lstm_size=32,
        lstm_num_layers=2,
        tanh_constant=2.5,
        temperature=5.0,
    ):
        super(Controller, self).__init__()
        # assign the attributes
        self.max_nodes = max_nodes
        self.num_edge = len(edge2index)
        self.edge2index = edge2index
        self.num_ops = len(op_names)
        self.op_names = op_names
        self.lstm_size = lstm_size
        self.lstm_N = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature
        # create parameters
        self.register_parameter(
            "input_vars", nn.Parameter(torch.Tensor(1, 1, lstm_size))
        )
        self.w_lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.lstm_N,
        )
        self.w_embd = nn.Embedding(self.num_ops, self.lstm_size)
        self.w_pred = nn.Linear(self.lstm_size, self.num_ops)

        nn.init.uniform_(self.input_vars, -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_embd.weight, -0.1, 0.1)
        nn.init.uniform_(self.w_pred.weight, -0.1, 0.1)

    def convert_structure(self, _arch):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                op_index = _arch[self.edge2index[node_str]]
                op_name = self.op_names[op_index]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def forward(self):

        inputs, h0 = self.input_vars, None
        log_probs, entropys, sampled_arch = [], [], []
        for iedge in range(self.num_edge):
            outputs, h0 = self.w_lstm(inputs, h0)

            logits = self.w_pred(outputs)
            logits = logits / self.temperature
            logits = self.tanh_constant * torch.tanh(logits)
            # distribution
            op_distribution = Categorical(logits=logits)
            op_index = op_distribution.sample()
            sampled_arch.append(op_index.item())

            op_log_prob = op_distribution.log_prob(op_index)
            log_probs.append(op_log_prob.view(-1))
            op_entropy = op_distribution.entropy()
            entropys.append(op_entropy.view(-1))

            # obtain the input embedding for the next step
            inputs = self.w_embd(op_index)
        return (
            torch.sum(torch.cat(log_probs)),
            torch.sum(torch.cat(entropys)),
            self.convert_structure(sampled_arch),
        )


class GenericNAS201Model(nn.Module):
    def __init__(
        self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats
    ):
        super(GenericNAS201Model, self).__init__()
        self._C = C
        self._layerN = N
        self._max_nodes = max_nodes
        self._stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
        C_prev, num_edge, edge2index = C, None, None
        self._cells = nn.ModuleList()
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
            self._cells.append(cell)
            C_prev = cell.out_dim
        self._op_names = deepcopy(search_space)
        self._Layer = len(self._cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(
            nn.BatchNorm2d(
                C_prev, affine=affine, track_running_stats=track_running_stats
            ),
            nn.ReLU(inplace=True),
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self._num_edge = num_edge
        # algorithm related
        self.arch_parameters = nn.Parameter(
            1e-3 * torch.randn(num_edge, len(search_space))
        )
        self._mode = None
        self.dynamic_cell = None
        self._tau = None
        self._algo = None
        self._drop_path = None
        self.verbose = False

    def set_algo(self, algo: Text):
        # used for searching
        assert self._algo is None, "This functioin can only be called once."
        self._algo = algo
        if algo == "enas":
            self.controller = Controller(
                self.edge2index, self._op_names, self._max_nodes
            )
        else:
            self.arch_parameters = nn.Parameter(
                1e-3 * torch.randn(self._num_edge, len(self._op_names))
            )
            if algo == "gdas":
                self._tau = 10

    def set_cal_mode(self, mode, dynamic_cell=None):
        assert mode in ["gdas", "enas", "urs", "joint", "select", "dynamic"]
        self._mode = mode
        if mode == "dynamic":
            self.dynamic_cell = deepcopy(dynamic_cell)
        else:
            self.dynamic_cell = None

    def set_drop_path(self, progress, drop_path_rate):
        if drop_path_rate is None:
            self._drop_path = None
        elif progress is None:
            self._drop_path = drop_path_rate
        else:
            self._drop_path = progress * drop_path_rate

    @property
    def mode(self):
        return self._mode

    @property
    def drop_path(self):
        return self._drop_path

    @property
    def weights(self):
        xlist = list(self._stem.parameters())
        xlist += list(self._cells.parameters())
        xlist += list(self.lastact.parameters())
        xlist += list(self.global_pooling.parameters())
        xlist += list(self.classifier.parameters())
        return xlist

    def set_tau(self, tau):
        self._tau = tau

    @property
    def tau(self):
        return self._tau

    @property
    def alphas(self):
        if self._algo == "enas":
            return list(self.controller.parameters())
        else:
            return [self.arch_parameters]

    @property
    def message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self._cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self._cells), cell.extra_repr()
            )
        return string

    def show_alphas(self):
        with torch.no_grad():
            if self._algo == "enas":
                return "w_pred :\n{:}".format(self.controller.w_pred.weight)
            else:
                return "arch-parameters :\n{:}".format(
                    nn.functional.softmax(self.arch_parameters, dim=-1).cpu()
                )

    def extra_repr(self):
        return "{name}(C={_C}, Max-Nodes={_max_nodes}, N={_layerN}, L={_Layer}, alg={_algo})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    @property
    def genotype(self):
        genotypes = []
        for i in range(1, self._max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                with torch.no_grad():
                    weights = self.arch_parameters[self.edge2index[node_str]]
                    op_name = self._op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def dync_genotype(self, use_random=False):
        genotypes = []
        with torch.no_grad():
            alphas_cpu = nn.functional.softmax(self.arch_parameters, dim=-1)
        for i in range(1, self._max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                if use_random:
                    op_name = random.choice(self._op_names)
                else:
                    weights = alphas_cpu[self.edge2index[node_str]]
                    op_index = torch.multinomial(weights, 1).item()
                    op_name = self._op_names[op_index]
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
                op_index = self._op_names.index(op)
                select_logits.append(logits[self.edge2index[node_str], op_index])
        return sum(select_logits).item()

    def return_topK(self, K, use_random=False):
        archs = Structure.gen_all(self._op_names, self._max_nodes, False)
        pairs = [(self.get_log_prob(arch), arch) for arch in archs]
        if K < 0 or K >= len(archs):
            K = len(archs)
        if use_random:
            return random.sample(archs, K)
        else:
            sorted_pairs = sorted(pairs, key=lambda x: -x[0])
            return_pairs = [sorted_pairs[_][1] for _ in range(K)]
            return return_pairs

    def normalize_archp(self):
        if self.mode == "gdas":
            while True:
                gumbels = -torch.empty_like(self.arch_parameters).exponential_().log()
                logits = (self.arch_parameters.log_softmax(dim=1) + gumbels) / self.tau
                probs = nn.functional.softmax(logits, dim=1)
                index = probs.max(-1, keepdim=True)[1]
                one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                hardwts = one_h - probs.detach() + probs
                if (
                    (torch.isinf(gumbels).any())
                    or (torch.isinf(probs).any())
                    or (torch.isnan(probs).any())
                ):
                    continue
                else:
                    break
            with torch.no_grad():
                hardwts_cpu = hardwts.detach().cpu()
            return hardwts, hardwts_cpu, index, "GUMBEL"
        else:
            alphas = nn.functional.softmax(self.arch_parameters, dim=-1)
            index = alphas.max(-1, keepdim=True)[1]
            with torch.no_grad():
                alphas_cpu = alphas.detach().cpu()
            return alphas, alphas_cpu, index, "SOFTMAX"

    def forward(self, inputs):
        alphas, alphas_cpu, index, verbose_str = self.normalize_archp()
        feature = self._stem(inputs)
        for i, cell in enumerate(self._cells):
            if isinstance(cell, SearchCell):
                if self.mode == "urs":
                    feature = cell.forward_urs(feature)
                    if self.verbose:
                        verbose_str += "-forward_urs"
                elif self.mode == "select":
                    feature = cell.forward_select(feature, alphas_cpu)
                    if self.verbose:
                        verbose_str += "-forward_select"
                elif self.mode == "joint":
                    feature = cell.forward_joint(feature, alphas)
                    if self.verbose:
                        verbose_str += "-forward_joint"
                elif self.mode == "dynamic":
                    feature = cell.forward_dynamic(feature, self.dynamic_cell)
                    if self.verbose:
                        verbose_str += "-forward_dynamic"
                elif self.mode == "gdas":
                    feature = cell.forward_gdas(feature, alphas, index)
                    if self.verbose:
                        verbose_str += "-forward_gdas"
                elif self.mode == "gdas_v1":
                    feature = cell.forward_gdas_v1(feature, alphas, index)
                    if self.verbose:
                        verbose_str += "-forward_gdas_v1"
                else:
                    raise ValueError("invalid mode={:}".format(self.mode))
            else:
                feature = cell(feature)
            if self.drop_path is not None:
                feature = drop_path(feature, self.drop_path)
        if self.verbose and random.random() < 0.001:
            print(verbose_str)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits
