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

import torch.nn as nn


def init_net(net, w_type, b_type):
    """
    Init network with various algorithms
    :param net:
    :param w_type:
    :param b_type:
    :return:
    """
    if w_type == 'none':
        pass
    elif w_type == 'xavier':
        net.apply(_init_weights_vs)
    elif w_type == 'kaiming':
        net.apply(_init_weights_he)
    elif w_type == 'zero':
        net.apply(_init_weights_zero)
    else:
        raise NotImplementedError(f'init_type={w_type} is not supported.')

    if b_type == 'none':
        pass
    elif b_type == 'xavier':
        net.apply(_init_bias_vs)
    elif b_type == 'kaiming':
        net.apply(_init_bias_he)
    elif b_type == 'zero':
        net.apply(_init_bias_zero)
    else:
        raise NotImplementedError(f'init_type={b_type} is not supported.')


def _init_weights_vs(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)


def _init_bias_vs(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if m.bias is not None:
            nn.init.xavier_normal_(m.bias)


def _init_weights_he(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight)


def _init_bias_he(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if m.bias is not None:
            nn.init.kaiming_normal_(m.bias)


def _init_weights_zero(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        m.weight.data.fill_(.0)


def _init_bias_zero(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if m.bias is not None:
            m.bias.data.fill_(.0)
