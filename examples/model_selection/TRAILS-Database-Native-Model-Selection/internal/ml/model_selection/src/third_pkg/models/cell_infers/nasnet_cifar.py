#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
import torch
import torch.nn as nn
from copy import deepcopy

from .cells import NASNetInferCell as InferCell, AuxiliaryHeadCIFAR


# The macro structure is based on NASNet
class NASNetonCIFAR(nn.Module):
    def __init__(
        self,
        C,
        N,
        stem_multiplier,
        num_classes,
        genotype,
        auxiliary,
        affine=True,
        track_running_stats=True,
    ):
        super(NASNetonCIFAR, self).__init__()
        self._C = C
        self._layerN = N
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

        C_prev_prev, C_prev, C_curr, reduction_prev = (
            C * stem_multiplier,
            C * stem_multiplier,
            C,
            False,
        )
        self.auxiliary_index = None
        self.auxiliary_head = None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            cell = InferCell(
                genotype,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
                affine,
                track_running_stats,
            )
            self.cells.append(cell)
            C_prev_prev, C_prev, reduction_prev = (
                C_prev,
                cell._multiplier * C_curr,
                reduction,
            )
            if reduction and C_curr == C * 4 and auxiliary:
                self.auxiliary_head = AuxiliaryHeadCIFAR(C_prev, num_classes)
                self.auxiliary_index = index
        self._Layer = len(self.cells)
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.drop_path_prob = -1

    def update_drop_path(self, drop_path_prob):
        self.drop_path_prob = drop_path_prob

    def auxiliary_param(self):
        if self.auxiliary_head is None:
            return []
        else:
            return list(self.auxiliary_head.parameters())

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_C}, N={_layerN}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def forward(self, inputs):
        stem_feature, logits_aux = self.stem(inputs), None
        cell_results = [stem_feature, stem_feature]
        for i, cell in enumerate(self.cells):
            cell_feature = cell(cell_results[-2], cell_results[-1], self.drop_path_prob)
            cell_results.append(cell_feature)
            if (
                self.auxiliary_index is not None
                and i == self.auxiliary_index
                and self.training
            ):
                logits_aux = self.auxiliary_head(cell_results[-1])
        out = self.lastact(cell_results[-1])
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        if logits_aux is None:
            return out, logits
        else:
            return out, [logits, logits_aux]
