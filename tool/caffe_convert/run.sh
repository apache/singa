#!/bin/bash
make

if [[ ! -f VGG_ILSVRC_19_layers.caffemodel ]]; then
    wget -c http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel
fi

python convert_model.py VGG-16.prototxt VGG_ILSVRC_19_layers.caffemodel vgg16

python vscaffe.py VGG-16.prototxt VGG_ILSVRC_16_layers.caffemodel vgg16
