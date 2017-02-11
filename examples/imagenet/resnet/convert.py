import os
import torchfile
import numpy as np
import cPickle as pickle
from argparse import ArgumentParser

'''Extract the net parameters from the torch file and store them as python dict
using cPickle'''

import model

verbose=False

def add_param(idx, name, val, params):
    if type(params) == dict:
        assert name not in params, 'duplicated param %s' % name
        params[name] = val
    else:
        assert params[idx].size() == val.size, 'size mismatch for %s: %s - %s' % (name, (params[idx].shape,), (val.shape,))
        params[idx].copy_from_numpy(val)

    if verbose:
        print name, val.shape


def conv(m, idx, params, param_names):
    outplane = m['weight'].shape[0]
    name = param_names[idx]
    val = np.reshape(m['weight'], (outplane, -1))
    add_param(idx, name, val, params)
    return idx + 1


def batchnorm(m, idx, params, param_names):
    add_param(idx, param_names[idx], m['weight'], params)
    add_param(idx + 1, param_names[idx + 1], m['bias'], params)
    add_param(idx + 2, param_names[idx + 2], m['running_mean'], params)
    add_param(idx + 3, param_names[idx + 3], m['running_var'], params)
    return idx + 4


def linear(m, idx, params, param_names):
    add_param(idx, param_names[idx], np.transpose(m['weight']), params)
    add_param(idx + 1, param_names[idx + 1], m['bias'], params)
    return idx + 2


def traverse(m, idx, params, param_names):
    ''' Traverse all modules of the torch checkpoint file to extract params.

    Args:
        m, a TorchObject
        idx, index for the current cursor of param_names
        params, an empty dictionary (name->numpy) to dump the params via pickle;
            or a list of tensor objects which should be in the same order as
            param_names, called to initialize net created in Singa directly
            using param values from torch checkpoint file.

    Returns:
        the updated idx
    '''
    module_type = m.__dict__['_typename']
    if module_type in ['nn.Sequential', 'nn.ConcatTable'] :
        for x in m.modules:
            idx = traverse(x, idx, params, param_names)
    elif 'SpatialConvolution' in module_type:
        idx = conv(m, idx, params, param_names)
    elif 'SpatialBatchNormalization' in module_type:
        idx = batchnorm(m, idx, params, param_names)
    elif 'Linear' in module_type:
        idx = linear(m, idx, params, param_names)
    return idx


if __name__ == '__main__':
    parser = ArgumentParser(description='Convert params from torch to python '
            'dict. \n resnet could have depth of 18, 34, 101, 152; \n
            wrn has depth 50; preact has depth 200; addbn has depth 50')
    parser.add_argument("infile", help="torch checkpoint file")
    parser.add_argument("model", choices = ['resnet', 'wrn', 'preact', 'addbn'])
    parser.add_argument("depth", type=int, choices = [18, 34, 50, 101, 152, 200])
    args = parser.parse_args()

    net = model.create_net(args.model, args.depth)
    # model.init_params(net)
    m = torchfile.load(args.infile)
    params = {}
    # params = net.param_values()
    param_names = net.param_names()
    traverse(m, 0, params, param_names)
    miss = [name for name in param_names if name not in params]
    if len(miss) > 0:
        print 'The following params are missing from torch file'
        print miss

    outfile = os.path.splitext(args.infile)[0] + '.pickle'
    with open(outfile, 'wb') as fd:
        pickle.dump(params, fd)
