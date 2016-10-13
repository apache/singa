#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import argparse,re
import caffe_parse.parse_from_protobuf as parse
from convert_symbol import proto2symbol
from singa import layer
from singa import initializer
from singa import metric
from singa import device
from singa import tensor
from singa import net

def get_iter(layers, net):
    '''
    Get param_names and param_values from convoluton and fully connect layer
    '''
    layer_with_p = []
    layer_with_p_type = []
    singa_p = []
    singa_wb = []
    for layer in layers:
        #print 'layer type:', layer.type
        if layer.type == 'Convolution' or layer.type == 'InnerProduct' or layer.type == 4 or layer.type == 14:
            layer_with_p_type.append(layer.type)
            layer_with_p.append(layer.blobs)

    for (p, v) in zip(net.param_specs(), net.param_values()):
        if p.name[:4] == 'conv' or p.name[0:2]== 'fc':
            singa_p.append(v)

    for (weight, bias) in zip(singa_p[0:len(singa_p):2], singa_p[1:len(singa_p):2]):
        singa_wb.append([weight,bias])

    assert (len(layer_with_p) == len(singa_wb))

    for (layer_type, blobs, wb) in zip(layer_with_p_type, layer_with_p, singa_wb):
        yield (layer_type, blobs, wb)


def main():
    parser = argparse.ArgumentParser(description='Caffe prototxt to singa model parameter converter.\
                    Note that only basic functions are implemented. You are welcomed to contribute to this file.')
    parser.add_argument('caffe_prototxt', help='The prototxt file in Caffe format')
    parser.add_argument('caffe_model', help='The binary model parameter file in Caffe format')
    parser.add_argument('save_model_name', help='The name of the output model prefix')
    args = parser.parse_args()

    # parser caffe_prototxt then transform to a Singa VGG-16 net
    net  = proto2symbol(args.caffe_prototxt)

    # call and don't call initializer the params name should be different maybe these is a bug. Now just call it.
    for (p, name) in zip(net.param_values(), net.param_names()):
        #print name, p.shape
        if 'mean' in name or 'beta' in name:
            p.set_value(0.0)
        elif 'var' in name:
            p.set_value(1.0)
        elif 'gamma' in name:
            initializer.uniform(p, 0, 1)
        elif len(p.shape) > 1:
            if 'conv' in name:
                initializer.gaussian(p, 0, 3 * 3 * p.shape[0])
            else:
                p.gaussian(0, 0.02)
        else:
            p.set_value(0)
        #print name, p.l1()

    # parser trained caffemodel
    layers = parse.parse_caffemodel(args.caffe_model)

    # get trained params
    iter = get_iter(layers, net)

    # now fill trained params to Singa VGG-16
    for layer_type, layer_blobs, wb in iter:
        assert(len(layer_blobs) == 2)
        wmat_dim = []
        if getattr(layer_blobs[0].shape, 'dim', None) is not None:
            if len(layer_blobs[0].shape.dim) > 0:
                wmat_dim = layer_blobs[0].shape.dim
            else:
                wmat_dim = [layer_blobs[0].num, layer_blobs[0].channels, layer_blobs[0].height, layer_blobs[0].width]
        else:
            wmat_dim = list(layer_blobs[0].shape)
        wmat = np.array(layer_blobs[0].data).reshape(wmat_dim)
        #print 'caffe wmat shape', wmat.shape
        wmat = wmat.astype(np.float32)
        #print 'caffe wmat:', wmat
        bias = np.array(layer_blobs[1].data)
        bias = bias.astype(np.float32)
        #print 'caffe bias shape', bias.shape
        #print 'caffe bias:', bias
        channels = layer_blobs[0].channels;
        params = wb
        params[0].copy_from_numpy(wmat)
        params[1].copy_from_numpy(bias)
        #print 'Elevent wmat', tensor.to_numpy(params[0])
        #print 'Elevent bias', tensor.to_numpy(params[1])

    net.save(args.save_model_name, 400)  # save model params into checkpoint file

if __name__ == '__main__':
    main()
