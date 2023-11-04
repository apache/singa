##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import math, torch
import torch.nn as nn


def select2withP(logits, tau, just_prob=False, num=2, eps=1e-7):
    if tau <= 0:
        new_logits = logits
        probs = nn.functional.softmax(new_logits, dim=1)
    else:
        while True:  # a trick to avoid the gumbels bug
            gumbels = -torch.empty_like(logits).exponential_().log()
            new_logits = (logits.log_softmax(dim=1) + gumbels) / tau
            probs = nn.functional.softmax(new_logits, dim=1)
            if (
                (not torch.isinf(gumbels).any())
                and (not torch.isinf(probs).any())
                and (not torch.isnan(probs).any())
            ):
                break

    if just_prob:
        return probs

    # with torch.no_grad(): # add eps for unexpected torch error
    #  probs = nn.functional.softmax(new_logits, dim=1)
    #  selected_index = torch.multinomial(probs + eps, 2, False)
    with torch.no_grad():  # add eps for unexpected torch error
        probs = probs.cpu()
        selected_index = torch.multinomial(probs + eps, num, False).to(logits.device)
    selected_logit = torch.gather(new_logits, 1, selected_index)
    selcted_probs = nn.functional.softmax(selected_logit, dim=1)
    return selected_index, selcted_probs


def ChannelWiseInter(inputs, oC, mode="v2"):
    if mode == "v1":
        return ChannelWiseInterV1(inputs, oC)
    elif mode == "v2":
        return ChannelWiseInterV2(inputs, oC)
    else:
        raise ValueError("invalid mode : {:}".format(mode))


def ChannelWiseInterV1(inputs, oC):
    assert inputs.dim() == 4, "invalid dimension : {:}".format(inputs.size())

    def start_index(a, b, c):
        return int(math.floor(float(a * c) / b))

    def end_index(a, b, c):
        return int(math.ceil(float((a + 1) * c) / b))

    batch, iC, H, W = inputs.size()
    outputs = torch.zeros((batch, oC, H, W), dtype=inputs.dtype, device=inputs.device)
    if iC == oC:
        return inputs
    for ot in range(oC):
        istartT, iendT = start_index(ot, oC, iC), end_index(ot, oC, iC)
        values = inputs[:, istartT:iendT].mean(dim=1)
        outputs[:, ot, :, :] = values
    return outputs


def ChannelWiseInterV2(inputs, oC):
    assert inputs.dim() == 4, "invalid dimension : {:}".format(inputs.size())
    batch, C, H, W = inputs.size()
    if C == oC:
        return inputs
    else:
        return nn.functional.adaptive_avg_pool3d(inputs, (oC, H, W))
    # inputs_5D = inputs.view(batch, 1, C, H, W)
    # otputs_5D = nn.functional.interpolate(inputs_5D, (oC,H,W), None, 'area', None)
    # otputs    = otputs_5D.view(batch, oC, H, W)
    # otputs_5D = nn.functional.interpolate(inputs_5D, (oC,H,W), None, 'trilinear', False)
    # return otputs


def linear_forward(inputs, linear):
    if linear is None:
        return inputs
    iC = inputs.size(1)
    weight = linear.weight[:, :iC]
    if linear.bias is None:
        bias = None
    else:
        bias = linear.bias
    return nn.functional.linear(inputs, weight, bias)


def get_width_choices(nOut):
    xsrange = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if nOut is None:
        return len(xsrange)
    else:
        Xs = [int(nOut * i) for i in xsrange]
        # xs = [ int(nOut * i // 10) for i in range(2, 11)]
        # Xs = [x for i, x in enumerate(xs) if i+1 == len(xs) or xs[i+1] > x+1]
        Xs = sorted(list(set(Xs)))
        return tuple(Xs)


def get_depth_choices(nDepth):
    if nDepth is None:
        return 3
    else:
        assert nDepth >= 3, "nDepth should be greater than 2 vs {:}".format(nDepth)
        if nDepth == 1:
            return (1, 1, 1)
        elif nDepth == 2:
            return (1, 1, 2)
        elif nDepth >= 3:
            return (nDepth // 3, nDepth * 2 // 3, nDepth)
        else:
            raise ValueError("invalid Depth : {:}".format(nDepth))


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = x.new_zeros(x.size(0), 1, 1, 1)
        mask = mask.bernoulli_(keep_prob)
        x = x * (mask / keep_prob)
        # x.div_(keep_prob)
        # x.mul_(mask)
    return x
