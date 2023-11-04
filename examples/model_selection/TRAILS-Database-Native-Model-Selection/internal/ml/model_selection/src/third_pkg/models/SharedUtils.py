#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
import torch
import torch.nn as nn


def additive_func(A, B):
    assert A.dim() == B.dim() and A.size(0) == B.size(0), "{:} vs {:}".format(
        A.size(), B.size()
    )
    C = min(A.size(1), B.size(1))
    if A.size(1) == B.size(1):
        return A + B
    elif A.size(1) < B.size(1):
        out = B.clone()
        out[:, :C] += A
        return out
    else:
        out = A.clone()
        out[:, :C] += B
        return out


def change_key(key, value):
    def func(m):
        if hasattr(m, key):
            setattr(m, key, value)

    return func


def parse_channel_info(xstring):
    blocks = xstring.split(" ")
    blocks = [x.split("-") for x in blocks]
    blocks = [[int(_) for _ in x] for x in blocks]
    return blocks
