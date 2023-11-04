#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
from typing import List, Text, Any
import torch.nn as nn

from ..cell_operations import ResNetBasicblock
from ..cell_infers.cells import InferCell


class DynamicShapeTinyNet(nn.Module):
    def __init__(self, channels: List[int], genotype: Any, num_classes: int):
        super(DynamicShapeTinyNet, self).__init__()
        self._channels = channels
        if len(channels) % 3 != 2:
            raise ValueError("invalid number of layers : {:}".format(len(channels)))
        self._num_stage = N = len(channels) // 3

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
        )

        # layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        c_prev = channels[0]
        self.cells = nn.ModuleList()
        for index, (c_curr, reduction) in enumerate(zip(channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(c_prev, c_curr, 2, True)
            else:
                cell = InferCell(genotype, c_prev, c_curr, 1)
            self.cells.append(cell)
            c_prev = cell.out_dim
        self._num_layer = len(self.cells)

        self.lastact = nn.Sequential(nn.BatchNorm2d(c_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)

    def get_message(self) -> Text:
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_channels}, N={_num_stage}, L={_num_layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def forward(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits
