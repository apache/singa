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
#include "singa/model/metric.h"
#include "singa/utils/channel.h"
#include "singa/utils/string.h"
#include "../../src/model/layer/cudnn_convolution.h"
#include "../../src/model/layer/cudnn_activation.h"
#include "../../src/model/layer/cudnn_pooling.h"
#include "../../src/model/layer/dense.h"
#include "../../src/model/layer/flatten.h"
namespace singa {

LayerConf GenConvConf(string name, int nb_filter, int kernel, int stride,
                      int pad) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("CudnnConvolution");
  ConvolutionConf *conv = conf.mutable_convolution_conf();
  conv->set_num_output(nb_filter);
  conv->add_kernel_size(kernel);
  conv->add_stride(stride);
  conv->add_pad(pad);

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
    pool->set_pool(PoolingConf_PoolMethod_AVE);
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
  DenseConf *dense = conf.mutable_dense_conf();
  dense->set_num_output(num_output);
  FillerConf *weight = dense->mutable_weight_filler();
  weight->set_type("Xavier");
  return conf;
}

LayerConf GenSoftmaxConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("CudnnSoftmax");
  return conf;
}

LayerConf GenFlattenConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("Flatten");
  return conf;
}
FeedForwardNet CreateNet(Optimizer* opt, Loss<Tensor>* loss, Metric<Tensor>* metric) {
  FeedForwardNet net;
  Shape s{3, 32, 32};

  net.Add(new CudnnConvolution(), GenConvConf("conv1", 32, 5, 1, 2), &s);
  net.Add(new CudnnActivation(), GenReLUConf("relu1"));
  net.Add(new CudnnPooling, GenPoolingConf("pool1", true, 3, 2, 0));
  net.Add(new CudnnConvolution(), GenConvConf("conv2", 32, 5, 1, 2));
  net.Add(new CudnnActivation(), GenReLUConf("relu2"));
  net.Add(new CudnnPooling(), GenPoolingConf("pool2", true, 3, 2, 0));
  net.Add(new CudnnConvolution, GenConvConf("conv3", 64, 5, 1, 2));
  net.Add(new CudnnActivation(), GenReLUConf("relu3"));
  net.Add(new CudnnConvolution(), GenConvConf("pool3", true, 3, 2, 0));
  net.Add(new Flatten(), GenFlattenConf("flat"));
  net.Add(new Dense(), GenDenseConf("ip1", 10));
  OptimizerConf opt_conf;
  opt_conf.set_momentum(0.9);
  opt->Setup(opt_conf);
  net.Compile(true, opt, loss, metric);
  return net;
}

void Train(float lr, int num_epoch, string data_dir) {
  Cifar10 data(data_dir);
  Tensor train_x, train_y, test_x, test_y;
  {
    auto train = data.ReadTrainData();
    size_t nsamples = train.first.shape(0);
    auto matx = Reshape(train.first, Shape{nsamples, train.first.Size() / nsamples});
    const auto mean = Average(matx, 0);
    SubRow(mean, &matx);
    train_x = Reshape(matx, train.first.shape());
    train_y = train.second;
    auto test = data.ReadTestData();
    nsamples = test.first.shape(0);
    auto maty = Reshape(test.first, Shape{nsamples, test.first.Size() / nsamples});
    SubRow(mean, &maty);
    test_x = Reshape(maty, test.first.shape());
    test_y = test.second;
  }
  LOG(ERROR) << "creating net";
  SoftmaxCrossEntropy loss;
  Accuracy acc;
  SGD sgd;
  sgd.SetLearningRateGenerator([lr](int step) {return lr;});
  auto net = CreateNet(&sgd, &loss, &acc);

  auto cuda = std::make_shared<CudaGPU>();
  net.ToDevice(cuda);

  train_x.ToDevice(cuda);
  train_y.ToDevice(cuda);
  net.Train(50, num_epoch, train_x, train_y); // test_x, test_y);
}


}

int main(int argc, char** argv) {
  singa::InitChannel(nullptr);
  int pos = singa::ArgPos(argc, argv, "-epoch");
  int nEpoch = 5;
  if (pos != -1)
    nEpoch = atoi(argv[pos + 1]);
  pos = singa::ArgPos(argc, argv, "-lr");
  float lr = 0.01;
  if (pos != -1)
    lr = atof(argv[pos + 1]);
  pos = singa::ArgPos(argc, argv, "-data");
  string data = "cifar-10-batches-bin";
  if (pos != -1)
    data = argv[pos + 1];

  LOG(INFO) << "Start training";
  singa::Train(lr, nEpoch, data);
  LOG(INFO) << "End training";
}
