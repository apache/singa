# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from .nasbench2_ops import *


def gen_searchcell_mask_from_arch_str(arch_str):
    nodes = arch_str.split('+') 
    nodes = [node[1:-1].split('|') for node in nodes]
    nodes = [[op_and_input.split('~')  for op_and_input in node] for node in nodes]

    keep_mask = []
    for curr_node_idx in range(len(nodes)):
            for prev_node_idx in range(curr_node_idx+1): 
                _op = [edge[0] for edge in nodes[curr_node_idx] if int(edge[1]) == prev_node_idx]
                assert len(_op) == 1, 'The arch string does not follow the assumption of 1 connection between two nodes.'
                for _op_name in OPS.keys():
                    keep_mask.append(_op[0] == _op_name)
    return keep_mask


def get_model_from_arch_str(arch_str, num_classes, bn=True, init_channels=16):
    keep_mask = gen_searchcell_mask_from_arch_str(arch_str)
    net = NAS201Model(arch_str=arch_str, num_classes=num_classes, use_bn=bn, keep_mask=keep_mask, stem_ch=init_channels)
    return net


def get_super_model(num_classes, use_bn=True):
    net = NAS201Model(arch_str=arch_str, num_classes=num_classes, use_bn=use_bn)
    return net


class NAS201Model(nn.Module):

    def __init__(self, arch_str, num_classes, use_bn=True, keep_mask=None, stem_ch=16):
        super(NAS201Model, self).__init__()
        self.arch_str=arch_str
        self.num_classes=num_classes
        self.use_bn= use_bn
        self.stem_ch = stem_ch

        self.stem = stem(out_channels=stem_ch, use_bn=use_bn)
        self.stack_cell1 = nn.Sequential(*[SearchCell(in_channels=stem_ch, out_channels=stem_ch, stride=1, affine=False, track_running_stats=False, use_bn=use_bn, keep_mask=keep_mask) for i in range(5)])
        self.reduction1 = reduction(in_channels=stem_ch, out_channels=stem_ch*2)
        self.stack_cell2 = nn.Sequential(*[SearchCell(in_channels=stem_ch*2, out_channels=stem_ch*2, stride=1, affine=False, track_running_stats=False, use_bn=use_bn, keep_mask=keep_mask) for i in range(5)])
        self.reduction2 = reduction(in_channels=stem_ch*2, out_channels=stem_ch*4)
        self.stack_cell3 = nn.Sequential(*[SearchCell(in_channels=stem_ch*4, out_channels=stem_ch*4, stride=1, affine=False, track_running_stats=False, use_bn=use_bn, keep_mask=keep_mask) for i in range(5)])
        self.top = top(in_dims=stem_ch*4, num_classes=num_classes, use_bn=use_bn)

    def forward(self, x):
        x = self.stem(x)        

        x = self.stack_cell1(x)
        x = self.reduction1(x)

        x = self.stack_cell2(x)
        x = self.reduction2(x)

        x = self.stack_cell3(x)

        x = self.top(x)
        return x
    
    def get_prunable_copy(self, bn=False):
        model_new = get_model_from_arch_str(self.arch_str, self.num_classes, use_bn=bn, init_channels=self.stem_ch)

        #TODO this is quite brittle and doesn't work with nn.Sequential when bn is different
        # it is only required to maintain initialization -- maybe init after get_punable_copy?
        model_new.load_state_dict(self.state_dict(), strict=False)
        model_new.train()

        return model_new
    

def get_arch_str_from_model(net):
    search_cell = net.stack_cell1[0].options
    keep_mask = net.stack_cell1[0].keep_mask
    num_nodes = net.stack_cell1[0].num_nodes

    nodes = []
    idx = 0
    for curr_node in range(num_nodes -1):
        edges = []
        for prev_node in range(curr_node+1): # n-1 prev nodes
            for _op_name in OPS.keys():
                if keep_mask[idx]:
                    edges.append(f'{_op_name}~{prev_node}')
                idx += 1
        node_str = '|'.join(edges)
        node_str = f'|{node_str}|'
        nodes.append(node_str) 
    arch_str = '+'.join(nodes)
    return arch_str


if __name__ == "__main__":
    arch_str = '|nor_conv_3x3~0|+|none~0|none~1|+|avg_pool_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|'
    
    n = get_model_from_arch_str(arch_str=arch_str, num_classes=10)
    print(n.stack_cell1[0])
    
    arch_str2 = get_arch_str_from_model(n)
    print(arch_str)
    print(arch_str2)
    print(f'Are the two arch strings same? {arch_str == arch_str2}')
