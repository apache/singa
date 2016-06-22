/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/
#include "./cifar10.h"
#include "singa/model/feed_forward_net.h"
#include "singa/model/optimizer.h"
#include "singa/model/initializer.h"

namespace singa {

LayerConf GenConvConf(string name, int nb_filter, int kernel, int stride,
                      int pad) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("CudnnConvolution");
  ConvolutionConf *conv = conf.mutable_convolution_conf();
  conv->set_num_output(nb_filter);
  conv->set_kernel_size(kernel);
  conv->set_stride(stride);
  conv->set_pad(pad);

  FillerConf *weight = conv->mutable_weight_filler();
  weight->set_type("Xavier");
  return conf;
}

LayerConf GenPoolingConf(string name, bool max_pool, int kernel, int stride, int pad) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("CudnnPooling");
  PoolingConf *pool = conf.mutable_pooling_conf();
  pool->set_kernel_size(kernel);
  pool->set_stride(stride);
  pool->set_pad(pad);
  if (!max_pool)
    pool->set_pool(PoolingConf_AVE);
  return conf;
}

LayerConf GenReLUConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("RELU");
  return conf;
}

LayerConf GenDenseConf(string name, int num_output) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("Dense");
  DenseConf *dense = conf->mutable_dense_conf();
  dense->set_num_output(num_output);
  FillerConf *weight = conv->mutable_weight_filler();
  weight->set_type("Xavier");
  return conf;
}

LayerConf GenSoftmaxConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("CudnnSoftmax");
  return conf;
}


FeedForwordNet CreateNet(Optimizer* opt, Loss* loss, Metric* metric) {
  FeedForwordNet net;
  Shape s{3, 32, 32};
  net.AddLayer(GenConvConf("conv1", 32, 5, 1, 2), &s);
  net.AddLayer(GenReLUConf("relu1"));
  net.AddLayer(GenConvConf("pool1", 3, 2, 0));
  net.AddLayer(GenConvConf("conv2", 32, 5, 1, 2));
  net.AddLayer(GenReLUConf("relu2"));
  net.AddLayer(GenConvConf("pool2", 3, 2, 0));
  net.AddLayer(GenConvConf("conv3", 64, 5, 1, 2));
  net.AddLayer(GenReLUConf("relu3"));
  net.AddLayer(GenConvConf("pool3", 3, 2, 0));
  net.AddLayer(GenDenseConf("ip1", 10));
  net.AddLayer(GenSoftmaxConf("softmax"));

  OptimizerConf opt_conf;
  opt_conf.set_momentum(0.9);
  opt->Setup(opt_conf);
  net.Compile(true, opt, loss, metric);
  return net;
}

void Train(float lr, int num_epoch, string data_dir) {
  SoftmaxCrossEntropy loss;
  Accuracy acc;
  SGD sgd;
  sgd.SetLearningRate([lr](int step) {return lr;});
  auto net = CreateNet(&opt, &loss, &metric);
  Cifar10 data(data_dir);
  Tensor train_x, tain_y, test_x, test_y;
  {
    auto train = data.ReadTrainData();
    const auto mean = Average(train.first, 0);
    train_x = SubRow(train.first, mean);
    auto test = data.ReadTestData();
    test_x = SubRow(test.first, mean);
    train_y = train.second;
    test_y = test.second;
  }
  net.Train(100, num_epoch, train_x, train_y, test_x, test_y);
}

int main(int argc, char** argv) {
  InitChannel();
  int pos = ArgPos(argc, argv, "-epoch");
  int nEpoch = 5;
  if (pos != -1)
    nEpoch = atoi(argv[pos + 1]);
  pos = ArgPos(argc, argv, "-lr");
  float lr = 0.01;
  if (pos != -1)
    lr = atof(argv[pos + 1]);
  pos = ArgPos(argc, argv, "-data");
  string data = "cifar-10-batch-bin";
  if (pos != -1)
    data = argv[pos + 1];

  LOG(INFO) << "Start training";
  Train(lr, nEpoch, data);
  LOG(INFO) << "End training";
}
}
