import torchfile
import numpy as np
import cPickle as pickle

def conv(m, name, params):
    outplane = m['weight'].shape[0]
    params[name + '-conv_weight'] = np.reshape(m['weight'], (outplane, -1))
    return params

def batchnorm(m, name, params):
    params[name + '-bn_gamma'] = m['weight']
    params[name + '-bn_beta'] = m['bias']
    params[name + '-bn_mean'] = m['running_mean']
    params[name + '-bn_var'] = m['running_var']
    return params

def block(m, name, params, has_identity):
    branch=m[0].modules[0].modules
    params = conv(branch[0], name + '-1', params)
    params = batchnorm(branch[1], name + '-1', params)
    params = conv(branch[3], name + '-2', params)
    params = batchnorm(branch[4], name + '-2', params)
    params = conv(branch[6], name + '-3', params)
    params = batchnorm(branch[7], name + '-3', params)
    if not has_identity:
        shortcut = m[0].modules[1].modules
        params = conv(shortcut[0], name + '-shortcut', params)
        params = batchnorm(shortcut[1], name + '-shortcut', params)
    return params

def stage(sid, m, num_blk, params):
    for i in range(num_blk):
        params = block(m[i].modules, 'stage%d-blk%d' % (sid, i), params, i!=0)
    return params

params = {}
model = torchfile.load('wrn-50-2.t7').modules
params = conv(model[0], 'input', params)
params = batchnorm(model[1], 'input', params)
params = stage(0, model[4].modules, 3, params)
params = stage(1, model[5].modules, 4, params)
params = stage(2, model[6].modules, 6, params)
params = stage(3, model[7].modules, 3, params)

params['dense_weight'] = np.transpose(model[10]['weight'])
params['dense_bias'] = model[10]['bias']
with open('wrn-50-2.pickle', 'wb') as fd:
    pickle.dump(params, fd)
