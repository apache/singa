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
# =============================================================================

from google.protobuf import text_format
import argparse
import re
import sys
import caffe_parse.caffe_pb2
from singa import layer
from singa import metric
from singa import loss
from singa import net as ffnet
from singa.proto import model_pb2

# Read caffe proto file


def readProtoNetFile(filepath):
    net_config = ''
    net_config = caffe_parse.caffe_pb2.NetParameter()
    return readProtoFile(filepath, net_config)


def readProtoSolverFile(filepath):
    solver_config = ''
    solver_config = caffe_parse.caffe_pb2.SolverParameter()
    return readProtoFile(filepath, solver_config)

# Read and parse caffe proto file


def readProtoFile(filepath, parser_object):
    file = open(filepath, "r")
    if not file:
        raise self.ProcessException("ERROR (" + filepath + ")!")
    # Merges an ASCII representation of a protocol message into a message.
    text_format.Merge(str(file.read()), parser_object)
    file.close()
    return parser_object


def net2script(net_file, input_sample_shape):
    net = readProtoNetFile(net_file)
    layer_conf = ''
    layer.engine = 'singacpp'
    flatten_id = 0
    if len(net.layer):
        layer_conf = net.layer
    elif len(net.layers):
        layer_conf = net.layers
    else:
        raise Exception('Invalid proto file.')

    net = ffnet.FeedForwardNet()
    for i in range(len(layer_conf)):
        if layer_conf[i].type == 'Data' or layer_conf[i].type == 5:
            continue
        elif layer_conf[i].type == 'SoftmaxWithLoss' or layer_conf[
                i].type == 21:
            net.loss = loss.SoftmaxCrossEntropy()
        elif layer_conf[i].type == 'EuclideanLoss' or layer_conf[i].type == 7:
            net.loss = loss.SquareError()
        elif layer_conf[i].type == 'Accuracy' or layer_conf[i].type == 1:
            net.metric = metric.Accuracy
        else:
            strConf = layer_conf[i].SerializeToString()
            conf = model_pb2.LayerConf()
            conf.ParseFromString(strConf)
            lyr = layer.Layer(conf.name, conf)
            if len(net.layers) == 0:
                lyr.setup(input_sample_shape)
            if layer_conf[i].type == 'InnerProduct' or layer_conf[i].type == 14:
                net.add(layer.Flatten('flat' + str(flatten_id)))
            net.add(lyr)

    #for i in range(len(net.layers)):
    #    print net.layers[i].name, '\t',  net.layers[i].get_output_sample_shape()


def main():
    parser = argparse.ArgumentParser(
        description='Caffe prototxt to SINGA model parameter converter.\
                    Note that only basic functions are implemented. You are welcomed to contribute to this file.')
    parser.add_argument(
        'net_prototxt', help='Path to the prototxt file for net in Caffe format')
    parser.add_argument(
        'input_sample_shape',
        help='The shape(in tuple) of input sample, example: 3 32 32',
        nargs='*')
    args = parser.parse_args()

    net2script(sys.argv[1], tuple(map(int, sys.argv[2:])))


if __name__ == '__main__':
    main()
