"""Builds the Pytorch computational graph.

Tensors flowing into a single vertex are added together for all vertices
except the output, which is concatenated instead. Tensors flowing out of input
are always added.

If interior edge channels don't match, drop the extra channels (channels are
guaranteed non-decreasing). Tensors flowing out of the input as always
projected instead.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

from .base_ops import *

import torch
import torch.nn as nn


class Cell(nn.Module):
    """
    Builds the model using the adjacency matrix and op labels specified. Channels
    controls the module output channel count but the interior channels are
    determined via equally splitting the channel count whenever there is a
    concatenation of Tensors.
    """
    def __init__(self, spec_matrix, ops_matrix, in_channels, out_channels, bn=True):
        super(Cell, self).__init__()

        self.matrix = spec_matrix
        self.ops = ops_matrix

        self.num_vertices = np.shape(self.matrix)[0]

        self.vertex_channels = compute_vertex_channels(in_channels, out_channels, self.matrix)

        # append operations of each node
        self.vertex_op = nn.ModuleList([None])
        for t in range(1, self.num_vertices - 1):
            op = OP_MAP[self.ops[t]](self.vertex_channels[t], self.vertex_channels[t], bn=bn)
            self.vertex_op.append(op)

        # project inout of each node count from 1
        self.input_projection = nn.ModuleList([None]) # init the ModuleList, None is the operation at node_index = 0
        for t in range(1, self.num_vertices):
            # if this node is connected, then 1 × 1 projections are used to scale channel counts
            if self.matrix[0, t]:
                self.input_projection.append(projection(in_channels, self.vertex_channels[t], bn=bn))
            else:
                self.input_projection.append(None) # add operation at node_index t if it is not connected to the input.

    def forward(self, x):
        tensors = [x]
        # concatenation to output
        out_concat = []

        # 1. compute the output of each node
        for t in range(1, self.num_vertices-1):

            fan_in = [truncate(tensors[src], self.vertex_channels[t]) # each input of the node t
                      for src in range(1, t) if self.matrix[src, t] # all preceding nodes of node t
                      ]

            # if this node is connected to the input node, then 1 × 1 projections are used to scale channel counts
            if self.matrix[0, t]:
                assert self.input_projection[t] is not None
                fan_in.append(self.input_projection[t](x))

            # perform operation on node
            # vertex_input = torch.stack(fan_in, dim=0).sum(dim=0)
            vertex_input = sum(fan_in)
            # vertex_input = sum(fan_in) / len(fan_in)
            assert self.vertex_op[t] is not None
            vertex_output = self.vertex_op[t](vertex_input)

            tensors.append(vertex_output)

            # if this node is connected to the output node, then concatenation
            if self.matrix[t, self.num_vertices-1]:
                out_concat.append(tensors[t])

        # 2. get final outputs of the cell
        # inout connect ot output directly
        if not out_concat:
            assert self.matrix[0, self.num_vertices-1] # assert output is connected to input directly
            assert self.input_projection[self.num_vertices-1] is not None
            outputs = self.input_projection[self.num_vertices-1](tensors[0]) # this is projection operation.
        else:
            # only 1 node in cell
            if len(out_concat) == 1:
                outputs = out_concat[0]
            else:
                outputs = torch.cat(out_concat, 1)

            if self.matrix[0, self.num_vertices-1]:
                assert self.input_projection[self.num_vertices-1] is not None
                outputs = outputs + self.input_projection[self.num_vertices-1](tensors[0])

        return outputs


class NasBench101Network(nn.Module):
    def __init__(self, spec, init_channels, num_stacks, num_modules_per_stack, num_labels, bn):
        super(NasBench101Network, self).__init__()
        self.layers = nn.ModuleList([])
        self.spec = spec
        in_channels = 3

        out_channels = init_channels
        num_stacks = num_stacks
        num_modules_per_stack = num_modules_per_stack
        num_labels = num_labels
        bn = bn

        # stem consisting of one 3 × 3 convolution with 128 output channels
        stem_conv = ConvBnRelu(in_channels, out_channels, 3, 1, 1, bn=bn)
        self.layers.append(stem_conv)

        in_channels = out_channels
        # We stack each cell 3 times, followed by a down-sampling layer, in which the image height and width are halved
        # via max-pooling and the channel count is doubled
        for stack_num in range(num_stacks):
            # stem layer's output skips down-sampling
            if stack_num > 0:
                down_sample = nn.MaxPool2d(kernel_size=2, stride=2)
                self.layers.append(down_sample)
                out_channels *= 2

            for module_num in range(num_modules_per_stack):
                cell = Cell(spec.matrix, spec.ops, in_channels, out_channels, bn=bn)
                self.layers.append(cell)
                in_channels = out_channels

        # self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_channels, num_labels)

        self._initialize_weights()

    def forward(self, x):
        # ints = []
        for _, layer in enumerate(self.layers):
            x = layer(x)
            # ints.append(x)

        # global_pooling
        out = torch.mean(x, (2, 3))
        # ints.append(out)
        # logits = self.classifier(out.view(out.size(0), -1))
        logits = self.classifier(out)
        return logits

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                if n == 0:
                    print(" n is 0, something wrong ")
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def reset_zero_grads(self):
        self.zero_grad()

    def get_weights(self):
        xlist = []
        for m in self.modules():
            xlist.append(m.parameters())
        return xlist

    def get_alphas(self):
        return [self.arch_parameters]


# copy from nasbench101
def projection(in_channels, out_channels, bn=True):
    """1x1 projection (as in ResNet) followed by batch normalization and ReLU."""
    return ConvBnRelu(in_channels, out_channels, 1, bn=bn)


# copy from nasbench101
def truncate(inputs_ori, channels):
    """
    :param inputs: input of the node
    :param channels: How many channel of the node
    :return:
    """
    """Slice the inputs to channels if necessary."""
    inputs = inputs_ori.clone()
    input_channels = inputs.size()[1]
    if input_channels < channels:
        raise ValueError('input channel < output channels for truncate')
    elif input_channels == channels:
        return inputs   # No truncation necessary
    else:
        # Truncation should only be necessary when channel division leads to
        # vertices with +1 channels. The input vertex should always be projected to
        # the minimum channel count.
        assert input_channels - channels == 1
        return inputs[:, :channels, :, :]


# copy from nasbench101
def compute_vertex_channels(in_channels, out_channels, matrix):
    """Computes the number of channels at every vertex.

    Given the input channels and output channels, this calculates the number of
    channels at each interior vertex. Interior vertices have the same number of
    channels as the max of the channels of the vertices it feeds into. The output
    channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to
    compensate.

    Returns:
        list of channel counts, in order of the vertices.
    """
    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices # channels of the input node
    vertex_channels[0] = in_channels
    vertex_channels[num_vertices - 1] = out_channels  # channels of the output node

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = out_channels // in_degree[num_vertices - 1]
    correction = out_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
      if matrix[v, num_vertices - 1]:
          vertex_channels[v] = interior_channels
          if correction:
              vertex_channels[v] += 1
              correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        assert vertex_channels[v] > 0

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in = final_fan_in + vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == out_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return vertex_channels


