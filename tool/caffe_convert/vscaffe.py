#!/usr/bin/env python
# encoding: utf-8
from singa import device
from singa import tensor
from singa import layer
from singa import initializer
from singa import metric
from singa import loss
from convert_symbol import proto2symbol
import caffe_parse.parse_from_protobuf as parse
import numpy as np
import argparse,re
import caffe

GPU_ID = int(0)

# predict in Singa
def predict(net, images, dev):
    y = net.predict(images)
    y.to_host()
    feat= tensor.to_numpy(y)
    return feat

# class for predict in Caffe
class CaffeFeature(object):
    def __init__(self, MODEL):
        self.net, self.transformer = self.initCaffe(MODEL)

    def extractFeatureFromFile(self, imgPath):
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', caffe.io.load_image(imgPath))
        out = self.net.forward()
        data0 = self.net.blobs['data'].data[...]
        featCaffe0 = self.net.blobs['fc8'].data[0]
        for k, v in self.net.blobs.items():
            print k, v.data.shape
        for feat_name in self.net.blobs.keys():
            #print '\n Caffe layer {} out: \n'.format(feat_name), self.net.blobs[feat_name].data[...]
            if feat_name == 'flatdata':
                flatdata = self.net.blobs[feat_name].data[...]
            if feat_name == 'fc6':
                fc6_out = self.net.blobs[feat_name].data[...]
            if feat_name == 'fc7':
                fc7_out = self.net.blobs[feat_name].data[...]
        #for k, v in self.net.params.items():
        #    print '\nCaffe trained params', k, v[0].data[...], '\n', v[1].data[...]
        return featCaffe0.copy(), data0 , flatdata, fc6_out.copy(), fc7_out.copy()

    def initCaffe(self, MODEL):
        MODEL_FILE, PRETRAINED = MODEL[:2]
        caffe.set_device(GPU_ID)
        caffe.set_mode_gpu()
        net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array([93.5940, 104.7624, 129.1863]))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))
        net.blobs['data'].reshape(1, 3, 224, 224)
        return net, transformer


def main():
    parser = argparse.ArgumentParser(description='Caffe prototxt to singa model parameter converter.\
                    Note that only basic functions are implemented. You are welcomed to contribute to this file.')
    parser.add_argument('caffe_prototxt', help='The prototxt file in Caffe format')
    parser.add_argument('caffe_model', help='The binary model parameter file in Caffe format')
    parser.add_argument('save_model_name', help='The name of the output model prefix')
    args = parser.parse_args()

    # give a picture and get ouput feature of fc7 layer in caffe
    tripletVggModel = [args.caffe_prototxt, args.caffe_model]
    caffe_api = CaffeFeature(tripletVggModel)
    featCaffe0, data0, flatdata, fc6_out, fc7_out  = caffe_api.extractFeatureFromFile('test.jpg')

    # parser caffe_prototxt then transform to a Singa VGG-16 net
    net = proto2symbol(args.caffe_prototxt)
    # call and don't call initializer the params name should be different maybe these is a bug. Now just call it.
    for (p, name) in zip(net.param_values(), net.param_names()):
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
    # load save model which contain trained params from Caffe VGG-16
    net.load(args.save_model_name, 400)


    # give a picture and get ouput feature of fc7 layer in Singa
    dev = device.create_cuda_gpu()
    net.to_device(dev)
    image = tensor.Tensor((1, 3, 224, 224), dev)
    image.copy_from_numpy(data0)
    featSinga = predict(net, image, dev)

    image.to_host()
    print 'featCaffe0: \n' , featCaffe0
    print 'featCaffe fc6: \n' , fc6_out
    print 'featCaffe fc7: \n' , fc7_out

    print 'featSinga : \n', featSinga
    #print '\n', (featCaffe0 -featSinga)

if __name__ == '__main__':
    main()
