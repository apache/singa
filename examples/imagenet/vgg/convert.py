# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Extract the net parameters from the pytorch file and store them
as python dict using cPickle. Must install pytorch.
'''

import torch.utils.model_zoo as model_zoo

import numpy as np
from argparse import ArgumentParser

import model

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def rename(pname):
    p1 = pname.find('/')
    p2 = pname.rfind('/')
    assert p1 != -1 and p2 != -1, 'param name = %s is not correct' % pname
    if 'gamma' in pname:
        suffix = 'weight'
    elif 'beta' in pname:
        suffix = 'bias'
    elif 'mean' in pname:
        suffix = 'running_mean'
    elif 'var' in pname:
        suffix = 'running_var'
    else:
        suffix = pname[p2 + 1:]
    return pname[p1+1:p2] + '.' + suffix


if __name__ == '__main__':
    parser = ArgumentParser(description='Convert params from torch to python'
                            'dict. ')
    parser.add_argument("depth", type=int, choices=[11, 13, 16, 19])
    parser.add_argument("outfile")
    parser.add_argument("--batchnorm", action='store_true',
                        help='use batchnorm or not')

    args = parser.parse_args()

    net = model.create_net(args.depth, 1000, args.batchnorm)
    url = 'vgg%d' % args.depth
    if args.batchnorm:
        url += '_bn'
    torch_dict = model_zoo.load_url(model_urls[url])
    params = {'SINGA_VERSION': 1101}
    # params = net.param_values()
    for pname, pval in zip(net.param_names(), net.param_values()):
        torch_name = rename(pname)
        if torch_name in torch_dict:
            ary = torch_dict[torch_name].numpy()
            ary = np.array(ary, dtype=np.float32)
            if len(ary.shape) == 4:
                params[pname] = np.reshape(ary, (ary.shape[0], -1))
            else:
                params[pname] = np.transpose(ary)
        else:
            print('param=%s is missing in the ckpt file' % pname)
        assert pval.shape == params[pname].shape,\
               'shape mismatch for %s' % pname

    with open(args.outfile, 'wb') as fd:
        pickle.dump(params, fd)
