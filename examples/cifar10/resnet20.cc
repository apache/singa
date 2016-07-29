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

#include <cmath>
#include "../../src/model/layer/cudnn_activation.h"
#include "../../src/model/layer/cudnn_convolution.h"
#include "../../src/model/layer/cudnn_dropout.h"
#include "../../src/model/layer/cudnn_lrn.h"
#include "../../src/model/layer/cudnn_pooling.h"
#include "../../src/model/layer/dense.h"
#include "../../src/model/layer/flatten.h"
#include "../../src/model/layer/cudnn_batchnorm.h"
#include "../../src/model/layer/cudnn_softmax.h"
#include "../../src/model/layer/split.h"
#include "../../src/model/layer/merge.h"
#include "singa/model/feed_forward_net.h"
#include "singa/model/initializer.h"
#include "singa/model/metric.h"
#include "singa/model/optimizer.h"
#include "singa/utils/channel.h"
#include "singa/utils/string.h"
#include "./cifar10.h"

namespace singa {

LayerConf GenConvConf(string name, int nb_filter, int kernel, int stride,
                      int pad, float std) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("CudnnConvolution");
  ConvolutionConf *conv = conf.mutable_convolution_conf();
  conv->set_num_output(nb_filter);
  conv->add_kernel_size(kernel);
  conv->add_stride(stride);
  conv->add_pad(pad);
  conv->set_bias_term(true);

  ParamSpec *wspec = conf.add_param();
  wspec->set_name(name + "_weight");
  auto wfill = wspec->mutable_filler();
  wfill->set_type("Gaussian");
  wfill->set_std(std);

  ParamSpec *bspec = conf.add_param();
  bspec->set_name(name + "_bias");
  bspec->set_lr_mult(2);
  //  bspec->set_decay_mult(0);
  return conf;
}

LayerConf GenPoolingConf(string name, bool max_pool, int kernel, int stride,
                         int pad) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("CudnnPooling");
  PoolingConf *pool = conf.mutable_pooling_conf();
  pool->set_kernel_size(kernel);
  pool->set_stride(stride);
  pool->set_pad(pad);
  if (!max_pool) pool->set_pool(PoolingConf_PoolMethod_AVE);
  return conf;
}

LayerConf GenReLUConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("RELU");
  return conf;
}

LayerConf GenDenseConf(string name, int num_output, float std, float wd) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("Dense");
  DenseConf *dense = conf.mutable_dense_conf();
  dense->set_num_output(num_output);

  ParamSpec *wspec = conf.add_param();
  wspec->set_name(name + "_weight");
  wspec->set_decay_mult(wd);
  auto wfill = wspec->mutable_filler();
  wfill->set_type("Gaussian");
  wfill->set_std(std);

  ParamSpec *bspec = conf.add_param();
  bspec->set_name(name + "_bias");
  bspec->set_lr_mult(2);
  bspec->set_decay_mult(0);

  return conf;
}

LayerConf GenFlattenConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("Flatten");
  return conf;
}

LayerConf GenDropoutConf(string name, float dropout_ratio) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("CudnnDropout");
  DropoutConf *dropout = conf.mutable_dropout_conf();
  dropout->set_dropout_ratio(dropout_ratio);
  return conf;
}

LayerConf GenBatchNormConf(string name, float std) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("BatchNorm");
  BatchNormConf *bn = conf.mutable_batchnorm_conf();
  bn->set_factor(0.9);

  ParamSpec *wspec = conf.add_param();
  wspec->set_name(name + "_weight");
  auto wfill = wspec->mutable_filler();
  wfill->set_type("Gaussian");
  wfill->set_std(std);

  ParamSpec *bspec = conf.add_param();
  bspec->set_name(name + "_bias");
  bspec->set_lr_mult(2);
  //  bspec->set_decay_mult(0);

  ParamSpec *meanspec = conf.add_param();
  meanspec->set_name(name + "_mean");
  auto meanfill = meanspec->mutable_filler();
  meanfill->set_type("Constant");
  meanfill->set_std(0);

  ParamSpec *varspec = conf.add_param();
  varspec->set_name(name + "_variance");
  auto varfill = varspec->mutable_filler();
  varfill->set_type("Gaussian");
  varfill->set_std(std);
  return conf;
}

LayerConf GenSplitConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("Split");
  return conf;
}

LayerConf GenMergeConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("Merge");
  return conf;
}

LayerConf GenSoftmaxConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("CudnnSoftmax");
  return conf;
}

Layer* BuildingBlock(FeedForwardNet& net, string layer_name, int nb_filter,
            int stride, float std, Layer* src) {
  Layer* split = net.Add(new Split(), GenSplitConf("split" + layer_name), src);
  Layer* bn_br1 = nullptr;
  if (stride > 1) {
    Layer* conv_br1 = net.Add(new CudnnConvolution(),
          GenConvConf("conv" + layer_name + "_br1", nb_filter, 1, stride, 0, std), split);
    bn_br1 = net.Add(new CudnnBatchNorm(), GenBatchNormConf("bn" + layer_name + "_br1", std), conv_br1);
  }
  Layer* conv1 = net.Add(new CudnnConvolution(),
          GenConvConf("conv" + layer_name + "_br2a", nb_filter, 3, stride, 1, std), split);
  Layer* bn1 = net.Add(new CudnnBatchNorm(),
               GenBatchNormConf("bn" + layer_name + "_br2a", std), conv1);
  Layer* relu1 = net.Add(new CudnnActivation(), GenReLUConf("relu" + layer_name + "_br2a"), bn1);
  Layer* conv2 = net.Add(new CudnnConvolution(),
              GenConvConf("conv" + layer_name + "_br2b", nb_filter, 3, 1, 1, std), relu1);
  Layer* bn2 = net.Add(new CudnnBatchNorm(), GenBatchNormConf("bn" + layer_name + "_br2b", std), conv2);
  if (stride > 1)
   return net.Add(new Merge(), GenMergeConf("merge" + layer_name), vector<Layer*>{bn_br1, bn2});
  else return net.Add(new Merge(), GenMergeConf("merge" + layer_name), vector<Layer*>{split, bn2});
}

FeedForwardNet CreateResNet() {
  FeedForwardNet net;
  Shape s{3, 32, 32};

  Layer* c1 = net.Add(new CudnnConvolution(), GenConvConf("conv1", 16, 3, 1, 1, 0.01), &s);
  Layer* bn1 = net.Add(new CudnnBatchNorm(), GenBatchNormConf("bn1", 0.01), c1);
  Layer* relu1 = net.Add(new CudnnActivation(), GenReLUConf("relu1"), bn1);
  /// feature map 16 * 32 * 32

  Layer* bl2a = BuildingBlock(net, "2a", 16, 1, 0.01, relu1);
  Layer* bl2b = BuildingBlock(net, "2b", 16, 1, 0.01, bl2a);
  Layer* bl2c = BuildingBlock(net, "2c", 16, 1, 0.01, bl2b);
  /// feature map 16 * 32 * 32

  Layer* bl3a = BuildingBlock(net, "3a", 32, 2, 0.01, bl2c);
  Layer* bl3b = BuildingBlock(net, "3b", 32, 1, 0.01, bl3a);
  Layer* bl3c = BuildingBlock(net, "3c", 32, 1, 0.01, bl3b);
  /// feature map 32 * 16 * 16

  Layer* bl4a = BuildingBlock(net, "4a", 64, 2, 0.01, bl3c);
  Layer* bl4b = BuildingBlock(net, "4b", 64, 1, 0.01, bl4a);
  Layer* bl4c = BuildingBlock(net, "4c", 64, 1, 0.01, bl4b);
  /// feature map 64 * 8 * 8

  Layer* p4 = net.Add(new CudnnPooling(), GenPoolingConf("pool4", false, 8, 1, 0), bl4c);
  Layer* flat = net.Add(new Flatten(), GenFlattenConf("flat"), p4);
  Layer* ip5 = net.Add(new Dense(), GenDenseConf("ip5", 10, 0.01, 1), flat);
  net.Add(new CudnnSoftmax(), GenSoftmaxConf("softmax"), ip5);
  
  /*
  /// print net graph
  LOG(INFO) << "==================================";
  for (size_t i = 0; i < net.layers().size(); i++) {
    Layer* tmp = net.layers().at(i);
    LOG(INFO) << "layer: " << tmp->name();
    LOG(INFO) << "layer type: " << net.layer_type().at(tmp);
    if (net.src().find(tmp) != net.src().end()) {
      vector<Layer*> tmpsrc = net.src().find(tmp)->second;
      for (size_t j = 0; j < tmpsrc.size(); j++)
        LOG(INFO) << "src layer: " << tmpsrc.at(j)->name();
    } else LOG(INFO) << "src layer: ";
    if (net.dst().find(tmp) != net.dst().end()) {
      vector<Layer*> tmpdst = net.dst().find(tmp)->second;
      for (size_t j = 0; j < tmpdst.size(); j++)
        LOG(INFO) << "dst layer: " << tmpdst.at(j)->name();
    } else LOG(INFO) << "dst layer: ";
    LOG(INFO) << "===================================";
  }*/
  return net;
}

void Train(float lr, int num_epoch, string data_dir) {
  Cifar10 data(data_dir);
  Tensor train_x, train_y, test_x, test_y;
  {
    auto train = data.ReadTrainData();
    size_t nsamples = train.first.shape(0);
    auto mtrain =
        Reshape(train.first, Shape{nsamples, train.first.Size() / nsamples});
    const Tensor& mean = Average(mtrain, 0);
    SubRow(mean, &mtrain);
    train_x = Reshape(mtrain, train.first.shape());
    train_y = train.second;
    auto test = data.ReadTestData();
    nsamples = test.first.shape(0);
    auto mtest =
        Reshape(test.first, Shape{nsamples, test.first.Size() / nsamples});
    SubRow(mean, &mtest);
    test_x = Reshape(mtest, test.first.shape());
    test_y = test.second;
  }
  CHECK_EQ(train_x.shape(0), train_y.shape(0));
  CHECK_EQ(test_x.shape(0), test_y.shape(0));
  LOG(INFO) << "Training samples = " << train_y.shape(0)
            << ", Test samples = " << test_y.shape(0);
  auto net = CreateResNet();
  SGD sgd;
  OptimizerConf opt_conf;
  opt_conf.set_momentum(0.9);
  auto reg = opt_conf.mutable_regularizer();
  reg->set_coefficient(0.004);
  sgd.Setup(opt_conf);
  sgd.SetLearningRateGenerator([lr](int step) {
    if (step <= 80)
      return static_cast<double>(lr);
    else if (step <= 120)
      return 0.1 * static_cast<double>(lr);
    else
      return 0.01 * static_cast<double>(lr);
  });

  SoftmaxCrossEntropy loss;
  Accuracy acc;
  net.Compile(true, &sgd, &loss, &acc);

  auto cuda = std::make_shared<CudaGPU>();
  net.ToDevice(cuda);
  train_x.ToDevice(cuda);
  train_y.ToDevice(cuda);
  test_x.ToDevice(cuda);
  test_y.ToDevice(cuda);
  net.Train(100, num_epoch, train_x, train_y, test_x, test_y);
}
}

int main(int argc, char **argv) {
  singa::InitChannel(nullptr);
  int pos = singa::ArgPos(argc, argv, "-epoch");
  int nEpoch = 1;
  if (pos != -1) nEpoch = atoi(argv[pos + 1]);
  pos = singa::ArgPos(argc, argv, "-lr");
  float lr = 0.001;
  if (pos != -1) lr = atof(argv[pos + 1]);
  pos = singa::ArgPos(argc, argv, "-data");
  string data = "cifar-10-batches-bin";
  if (pos != -1) data = argv[pos + 1];

  LOG(INFO) << "Start training";
  singa::Train(lr, nEpoch, data);
  LOG(INFO) << "End training";
}

