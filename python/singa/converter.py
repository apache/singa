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
from singa import layer
from singa import metric
from singa import loss
from singa import net as ffnet
from .proto import model_pb2
from .proto import caffe_pb2


class CaffeConverter:

    def __init__(self, net_proto, solver_proto = None, input_sample_shape = None):
        self.caffe_net_path = net_proto
        self.caffe_solver_path = solver_proto
        self.input_sample_shape = input_sample_shape

    def read_net_proto(self):
        net_config = caffe_pb2.NetParameter()
        return self.read_proto(self.caffe_net_path, net_config)

    def read_solver_proto(self):
        solver_config = caffe_pb2.SolverParameter()
        return self.read_proto(self.caffe_solver_path, solver_config)

    def read_proto(self, filepath, parser_object):
        file = open(filepath, "r")
        if not file:
            raise self.ProcessException("ERROR (" + filepath + ")!")
        # Merges an ASCII representation of a protocol message into a message.
        text_format.Merge(str(file.read()), parser_object)
        file.close()
        return parser_object

    def convert_engine(self, layer_conf, solver_mode):
        '''
        Convert caffe engine into singa engine
        return:
            a singa engine string
        '''
        caffe_engine = ''
        singa_engine = ''

        # if no 'engine' field in caffe proto, set engine to -1
        if layer_conf.type == 'Convolution' or layer_conf.type == 4:
            caffe_engine = layer_conf.convolution_param.engine
        elif layer_conf.type == 'Pooling' or layer_conf.type == 17:
            caffe_engine = layer_conf.pooling_param.engine
        elif layer_conf.type == 'ReLU' or layer_conf.type == 18:
            caffe_engine = layer_conf.relu_param.engine
        elif layer_conf.type == 'Sigmoid' or layer_conf.type == 19:
            caffe_engine = layer_conf.sigmoid_param.engine
        elif layer_conf.type == 'TanH' or layer_conf.type == 23:
            caffe_engine = layer_conf.tanh_param.engine
        elif layer_conf.type == 'LRN' or layer_conf.type == 15:
            caffe_engine = layer_conf.lrn_param.engine
        elif layer_conf.type == 'Softmax' or layer_conf.type == 20:
            caffe_engine = layer_conf.softmax_param.engine
        elif layer_conf.type == 'InnerProduct' or layer_conf.type == 14:
            caffe_engine = -1
        elif layer_conf.type == 'Dropout' or layer_conf.type == 6:
            caffe_engine = -1
        elif layer_conf.type == 'Flatten' or layer_conf.type == 8:
            caffe_engine = -1
        else:
            raise Exception('Unknown layer type: ' + layer_conf.type)

        # caffe_engine: -1-no field;  0-DEFAULT; 1-CAFFE; 2-CUDNN
        # solver_mode: 0-CPU; 1-GPU
        if solver_mode == 1:
            singa_engine = 'cudnn'
        else:
            if caffe_engine == 2:
                raise Exception('engine and solver mode mismatch!')
            else:
                singa_engine = 'singacpp'

        if ((layer_conf.type == 'InnerProduct' or layer_conf.type == 14) or \
            (layer_conf.type == 'Flatten' or layer_conf.type == 8)) and \
            singa_engine == 'cudnn':
            singa_engine = 'singacuda'

        return singa_engine


    def create_net(self):
        '''
        Create singa net based on caffe proto files.
            net_proto: caffe prototxt that describes net
            solver_proto: caffe prototxt that describe solver
            input_sample_shape: shape of input data tensor
        return:
            a FeedForwardNet object
        '''
        caffe_net = self.read_net_proto()
        if self.caffe_solver_path is not None:
            caffe_solver = self.read_solver_proto()
        layer_confs = ''
        flatten_id = 0

        # If the net proto has the input shape
        if len(caffe_net.input_dim) > 0:
            self.input_sample_shape = caffe_net.input_dim
        if len(caffe_net.layer):
            layer_confs = caffe_net.layer
        elif len(caffe_net.layers):
            layer_confs = caffe_net.layers
        else:
            raise Exception('Invalid proto file!')

        net = ffnet.FeedForwardNet()
        for i in range(len(layer_confs)):
            if layer_confs[i].type == 'Data' or layer_confs[i].type == 5:
                continue
            elif layer_confs[i].type == 'SoftmaxWithLoss' or layer_confs[i].type == 21:
                net.loss = loss.SoftmaxCrossEntropy()
            elif layer_confs[i].type == 'EuclideanLoss' or layer_confs[i].type == 7:
                net.loss = loss.SquareError()
            elif layer_confs[i].type == 'Accuracy' or layer_confs[i].type == 1:
                net.metric = metric.Accuracy()
            else:
                strConf = layer_confs[i].SerializeToString()
                conf = model_pb2.LayerConf()
                conf.ParseFromString(strConf)
                if caffe_solver:
                    layer.engine = self.convert_engine(
                        layer_confs[i], caffe_solver.solver_mode)
                else:
                    layer.engine = self.convert_engine(layer_confs[i], 0)
                lyr = layer.Layer(conf.name, conf)
                if len(net.layers) == 0:
                    lyr.setup(self.input_sample_shape)
                    print lyr.name, lyr.get_output_sample_shape()
                if layer_confs[i].type == 'InnerProduct' or layer_confs[i].type == 14:
                    net.add(layer.Flatten('flat' + str(flatten_id)))
                    flatten_id += 1
                net.add(lyr)

        return net
