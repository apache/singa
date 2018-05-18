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

URL_PREFIX = 'https://download.pytorch.org/models/'
model_urls = {
    'densenet121': URL_PREFIX + 'densenet121-a639ec97.pth',
    'densenet169': URL_PREFIX + 'densenet169-b2777c0a.pth',
    'densenet201': URL_PREFIX + 'densenet201-c1103571.pth',
    'densenet161': URL_PREFIX + 'densenet161-8d451a50.pth',
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
    parser.add_argument("depth", type=int, choices=[121, 169, 201, 161])
    parser.add_argument("outfile")
    parser.add_argument('nb_classes', default=1000, type=int)

    args = parser.parse_args()

    net = model.create_net(args.depth, args.nb_classes)
    url = 'densenet%d' % args.depth
    torch_dict = model_zoo.load_url(model_urls[url])
    params = {'SINGA_VERSION': 1101}

    # resolve dict keys name mismatch problem
    print(len(net.param_names()), len(torch_dict.keys()))
    for pname, pval, torch_name in\
        zip(net.param_names(), net.param_values(), torch_dict.keys()):
        #torch_name = rename(pname)
        ary = torch_dict[torch_name].numpy()
        ary = np.array(ary, dtype=np.float32)
        if len(ary.shape) == 4:
            params[pname] = np.reshape(ary, (ary.shape[0], -1))
        else:
            params[pname] = np.transpose(ary)
        #pdb.set_trace()
        assert pval.shape == params[pname].shape, 'shape mismatch for {0}, \
               expected {1} in torch model, got {2} in singa model'.\
               format(pname, params[pname].shape, pval.shape)

    with open(args.outfile, 'wb') as fd:
        pickle.dump(params, fd)