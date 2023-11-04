#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
# Here, we utilized three techniques to search for the number of channels:
# - channel-wise interpolation from "Network Pruning via Transformable Architecture Search, NeurIPS 2019"
# - masking + Gumbel-Softmax (mask_gumbel) from "FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions, CVPR 2020"
# - masking + sampling (mask_rl) from "Can Weight Sharing Outperform Random Architecture Search? An Investigation With TuNAS, CVPR 2020"
from typing import List, Text, Any
import random, torch
import torch.nn as nn

from ..cell_operations import ResNetBasicblock
from ..cell_infers.cells import InferCell
from .SoftSelect import select2withP, ChannelWiseInter


class GenericNAS301Model(nn.Module):
    def __init__(
        self,
        candidate_Cs: List[int],
        max_num_Cs: int,
        genotype: Any,
        num_classes: int,
        affine: bool,
        track_running_stats: bool,
    ):
        super(GenericNAS301Model, self).__init__()
        self._max_num_Cs = max_num_Cs
        self._candidate_Cs = candidate_Cs
        if max_num_Cs % 3 != 2:
            raise ValueError("invalid number of layers : {:}".format(max_num_Cs))
        self._num_stage = N = max_num_Cs // 3
        self._max_C = max(candidate_Cs)

        stem = nn.Sequential(
            nn.Conv2d(3, self._max_C, kernel_size=3, padding=1, bias=not affine),
            nn.BatchNorm2d(
                self._max_C, affine=affine, track_running_stats=track_running_stats
            ),
        )

        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        c_prev = self._max_C
        self._cells = nn.ModuleList()
        self._cells.append(stem)
        for index, reduction in enumerate(layer_reductions):
            if reduction:
                cell = ResNetBasicblock(c_prev, self._max_C, 2, True)
            else:
                cell = InferCell(
                    genotype, c_prev, self._max_C, 1, affine, track_running_stats
                )
            self._cells.append(cell)
            c_prev = cell.out_dim
        self._num_layer = len(self._cells)

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(
                c_prev, affine=affine, track_running_stats=track_running_stats
            ),
            nn.ReLU(inplace=True),
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)
        # algorithm related
        self.register_buffer("_tau", torch.zeros(1))
        self._algo = None
        self._warmup_ratio = None

    def set_algo(self, algo: Text):
        # used for searching
        assert self._algo is None, "This functioin can only be called once."
        assert algo in ["mask_gumbel", "mask_rl", "tas"], "invalid algo : {:}".format(
            algo
        )
        self._algo = algo
        self._arch_parameters = nn.Parameter(
            1e-3 * torch.randn(self._max_num_Cs, len(self._candidate_Cs))
        )
        # if algo == 'mask_gumbel' or algo == 'mask_rl':
        self.register_buffer(
            "_masks", torch.zeros(len(self._candidate_Cs), max(self._candidate_Cs))
        )
        for i in range(len(self._candidate_Cs)):
            self._masks.data[i, : self._candidate_Cs[i]] = 1

    @property
    def tau(self):
        return self._tau

    def set_tau(self, tau):
        self._tau.data[:] = tau

    @property
    def warmup_ratio(self):
        return self._warmup_ratio

    def set_warmup_ratio(self, ratio: float):
        self._warmup_ratio = ratio

    @property
    def weights(self):
        xlist = list(self._cells.parameters())
        xlist += list(self.lastact.parameters())
        xlist += list(self.global_pooling.parameters())
        xlist += list(self.classifier.parameters())
        return xlist

    @property
    def alphas(self):
        return [self._arch_parameters]

    def show_alphas(self):
        with torch.no_grad():
            return "arch-parameters :\n{:}".format(
                nn.functional.softmax(self._arch_parameters, dim=-1).cpu()
            )

    @property
    def random(self):
        cs = []
        for i in range(self._max_num_Cs):
            index = random.randint(0, len(self._candidate_Cs) - 1)
            cs.append(str(self._candidate_Cs[index]))
        return ":".join(cs)

    @property
    def genotype(self):
        cs = []
        for i in range(self._max_num_Cs):
            with torch.no_grad():
                index = self._arch_parameters[i].argmax().item()
                cs.append(str(self._candidate_Cs[index]))
        return ":".join(cs)

    def get_message(self) -> Text:
        string = self.extra_repr()
        for i, cell in enumerate(self._cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self._cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(candidates={_candidate_Cs}, num={_max_num_Cs}, N={_num_stage}, L={_num_layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def forward(self, inputs):
        feature = inputs

        log_probs = []
        for i, cell in enumerate(self._cells):
            feature = cell(feature)
            # apply different searching algorithms
            idx = max(0, i - 1)
            if self._warmup_ratio is not None:
                if random.random() < self._warmup_ratio:
                    mask = self._masks[-1]
                else:
                    mask = self._masks[random.randint(0, len(self._masks) - 1)]
                feature = feature * mask.view(1, -1, 1, 1)
            elif self._algo == "mask_gumbel":
                weights = nn.functional.gumbel_softmax(
                    self._arch_parameters[idx : idx + 1], tau=self.tau, dim=-1
                )
                mask = torch.matmul(weights, self._masks).view(1, -1, 1, 1)
                feature = feature * mask
            elif self._algo == "tas":
                selected_cs, selected_probs = select2withP(
                    self._arch_parameters[idx : idx + 1], self.tau, num=2
                )
                with torch.no_grad():
                    i1, i2 = selected_cs.cpu().view(-1).tolist()
                c1, c2 = self._candidate_Cs[i1], self._candidate_Cs[i2]
                out_channel = max(c1, c2)
                out1 = ChannelWiseInter(feature[:, :c1], out_channel)
                out2 = ChannelWiseInter(feature[:, :c2], out_channel)
                out = out1 * selected_probs[0, 0] + out2 * selected_probs[0, 1]
                if feature.shape[1] == out.shape[1]:
                    feature = out
                else:
                    miss = torch.zeros(
                        feature.shape[0],
                        feature.shape[1] - out.shape[1],
                        feature.shape[2],
                        feature.shape[3],
                        device=feature.device,
                    )
                    feature = torch.cat((out, miss), dim=1)
            elif self._algo == "mask_rl":
                prob = nn.functional.softmax(
                    self._arch_parameters[idx : idx + 1], dim=-1
                )
                dist = torch.distributions.Categorical(prob)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                mask = self._masks[action.item()].view(1, -1, 1, 1)
                feature = feature * mask
            else:
                raise ValueError("invalid algorithm : {:}".format(self._algo))

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits, log_probs
