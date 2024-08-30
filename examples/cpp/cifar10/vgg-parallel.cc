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
#include <memory>
#include <thread>

#include "cifar10.h"
#include "singa/core/memory.h"
#include "singa/model/feed_forward_net.h"
#include "singa/model/initializer.h"
#include "singa/model/metric.h"
#include "singa/model/optimizer.h"
#include "singa/model/updater.h"
#include "singa/utils/channel.h"
#include "singa/utils/string.h"

namespace singa {

// currently supports 'cudnn' and 'singacpp'
const std::string engine = "cudnn";
const float default_wd = 0.0005f;

LayerConf GenConvConf(string name, int nb_filter, int kernel, int stride,
                      int pad, float std = .02f, float bias = .0f) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type(engine + "_convolution");
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
  wfill->set_std(sqrt(2.0f / (nb_filter * 9.0f)));

  ParamSpec *bspec = conf.add_param();
  bspec->set_name(name + "_bias");
  auto bfill = bspec->mutable_filler();
  bfill->set_value(bias);
  //  bspec->set_lr_mult(2);
  //  bspec->set_decay_mult(0);
  return conf;
}

LayerConf GenPoolingConf(string name, bool max_pool, int kernel, int stride,
                         int pad) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type(engine + "_pooling");
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
  conf.set_type(engine + "_relu");
  return conf;
}

LayerConf GenDenseConf(string name, int num_output, float std,
                       float wd = default_wd) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("singa_dense");
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
  conf.set_type("singa_flatten");
  return conf;
}

LayerConf GenBatchNormConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type(engine + "_batchnorm");
  ParamSpec *gammaspec = conf.add_param();
  gammaspec->set_name(name + "_gamma");
  auto gammafill = gammaspec->mutable_filler();
  gammafill->set_type("uniform");
  gammafill->set_min(0);
  gammafill->set_max(1);

  ParamSpec *betaspec = conf.add_param();
  betaspec->set_name(name + "_beta");
  auto betafill = betaspec->mutable_filler();
  betafill->set_type("constant");
  betafill->set_value(0);

  ParamSpec *meanspec = conf.add_param();
  meanspec->set_name(name + "_mean");
  auto meanfill = meanspec->mutable_filler();
  meanfill->set_type("constant");
  meanfill->set_value(0);

  ParamSpec *varspec = conf.add_param();
  varspec->set_name(name + "_var");
  auto varfill = varspec->mutable_filler();
  varfill->set_type("constant");
  varfill->set_value(1);

  return conf;
}

LayerConf GenDropoutConf(string name, float dropout_ratio) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type(engine + "_dropout");
  DropoutConf *dropout = conf.mutable_dropout_conf();
  dropout->set_dropout_ratio(dropout_ratio);

  return conf;
}

void ConvBNReLU(FeedForwardNet &net, string name, int nb_filter,
                Shape *shape = nullptr) {
  net.Add(GenConvConf(name + "_conv", nb_filter, 3, 1, 1), shape);
  net.Add(GenBatchNormConf(name + "_bn"));
  net.Add(GenReLUConf(name + "_relu"));
}

FeedForwardNet CreateNet() {
  FeedForwardNet net;
  Shape s{3, 32, 32};
  ConvBNReLU(net, "conv1_1", 64, &s);
  net.Add(GenDropoutConf("drop1", 0.3));
  ConvBNReLU(net, "conv1_2", 64);
  net.Add(GenPoolingConf("pool1", true, 2, 2, 0));
  ConvBNReLU(net, "conv2_1", 128);
  net.Add(GenDropoutConf("drop2", 0.4));
  ConvBNReLU(net, "conv2_2", 128);
  net.Add(GenPoolingConf("pool2", true, 2, 2, 0));
  ConvBNReLU(net, "conv3_1", 256);
  net.Add(GenDropoutConf("drop3_1", 0.4));
  ConvBNReLU(net, "conv3_2", 256);
  net.Add(GenDropoutConf("drop3_2", 0.4));
  ConvBNReLU(net, "conv3_3", 256);
  net.Add(GenPoolingConf("pool3", true, 2, 2, 0));
  ConvBNReLU(net, "conv4_1", 512);
  net.Add(GenDropoutConf("drop4_1", 0.4));
  ConvBNReLU(net, "conv4_2", 512);
  net.Add(GenDropoutConf("drop4_2", 0.4));
  ConvBNReLU(net, "conv4_3", 512);
  net.Add(GenPoolingConf("pool4", true, 2, 2, 0));
  ConvBNReLU(net, "conv5_1", 512);
  net.Add(GenDropoutConf("drop5_1", 0.4));
  ConvBNReLU(net, "conv5_2", 512);
  net.Add(GenDropoutConf("drop5_2", 0.4));
  ConvBNReLU(net, "conv5_3", 512);
  net.Add(GenPoolingConf("pool5", true, 2, 2, 0));
  net.Add(GenFlattenConf("flat"));
  net.Add(GenDropoutConf("flat_drop", 0.5));
  net.Add(GenDenseConf("ip1", 512, 0.02));
  net.Add(GenBatchNormConf("ip1_bn"));
  net.Add(GenReLUConf("ip1_relu"));
  net.Add(GenDropoutConf("ip1_drop", 0.5));
  net.Add(GenDenseConf("ip2", 10, 0.02));

  return net;
}

void Train(float lr, int num_epoch, string data_dir) {
  Cifar10 data(data_dir);
  Tensor train_x, train_y, test_x, test_y;
  Tensor train_x_1, train_x_2, train_y_1, train_y_2;
  {
    auto train = data.ReadTrainData();
    size_t nsamples = train.first.shape(0);
    auto mtrain =
        Reshape(train.first, Shape{nsamples, train.first.Size() / nsamples});
    const Tensor &mean = Average(mtrain, 0);
    SubRow(mean, &mtrain);
    Tensor std = Square(mtrain);
    std = Average(std, 0);
    std = Sqrt(std);
    ;
    std += 1e-6f;
    DivRow(std, &mtrain);

    train_x = Reshape(mtrain, train.first.shape());
    train_y = train.second;

    LOG(INFO) << "Slicing training data...";
    train_x_1 = Tensor(Shape{nsamples / 2, train.first.shape(1),
                             train.first.shape(2), train.first.shape(3)});
    LOG(INFO) << "Copying first data slice...";
    CopyDataToFrom(&train_x_1, train_x, train_x.Size() / 2);
    train_x_2 = Tensor(Shape{nsamples / 2, train.first.shape(1),
                             train.first.shape(2), train.first.shape(3)});
    LOG(INFO) << "Copying second data slice...";
    CopyDataToFrom(&train_x_2, train_x, train_x.Size() / 2, 0,
                   train_x.Size() / 2);
    train_y_1 = Tensor(Shape{nsamples / 2});
    train_y_1.AsType(kInt);
    LOG(INFO) << "Copying first label slice...";
    CopyDataToFrom(&train_y_1, train_y, train_y.Size() / 2);
    train_y_2 = Tensor(Shape{nsamples / 2});
    train_y_2.AsType(kInt);
    LOG(INFO) << "Copying second label slice...";
    CopyDataToFrom(&train_y_2, train_y, train_y.Size() / 2, 0,
                   train_y.Size() / 2);

    auto test = data.ReadTestData();
    nsamples = test.first.shape(0);
    auto mtest =
        Reshape(test.first, Shape{nsamples, test.first.Size() / nsamples});
    SubRow(mean, &mtest);
    DivRow(std, &mtest);
    test_x = Reshape(mtest, test.first.shape());
    test_y = test.second;
  }

  CHECK_EQ(train_x.shape(0), train_y.shape(0));
  CHECK_EQ(test_x.shape(0), test_y.shape(0));
  LOG(INFO) << "Total Training samples = " << train_y.shape(0)
            << ", Total Test samples = " << test_y.shape(0);
  CHECK_EQ(train_x_1.shape(0), train_y_1.shape(0));
  LOG(INFO) << "On net 1, Training samples = " << train_y_1.shape(0)
            << ", Test samples = " << test_y.shape(0);
  CHECK_EQ(train_x_2.shape(0), train_y_2.shape(0));
  LOG(INFO) << "On net 2, Training samples = " << train_y_2.shape(0);

  auto net_1 = CreateNet();
  auto net_2 = CreateNet();

  SGD sgd;
  OptimizerConf opt_conf;
  opt_conf.set_momentum(0.9);
  auto reg = opt_conf.mutable_regularizer();
  reg->set_coefficient(0.0005);
  sgd.Setup(opt_conf);
  sgd.SetLearningRateGenerator([lr](int epoch) {
    return 0.01f / static_cast<float>(1u << (epoch / 30));
  });

  SoftmaxCrossEntropy loss_1, loss_2;
  Accuracy acc_1, acc_2;
  /// Create updater aggregating gradient on CPU
  std::shared_ptr<Updater> updater = std::make_shared<LocalUpdater>(2, &sgd);

  /// Only need to register parameter once.
  net_1.Compile(true, true, updater, &loss_1, &acc_1);
  net_2.Compile(true, false, updater, &loss_2, &acc_2);

  MemPoolConf mem_conf;
  mem_conf.add_device(0);
  mem_conf.add_device(1);
  std::shared_ptr<DeviceMemPool> mem_pool(new CnMemPool(mem_conf));
  std::shared_ptr<CudaGPU> dev_1(new CudaGPU(0, mem_pool));
  std::shared_ptr<CudaGPU> dev_2(new CudaGPU(1, mem_pool));
  net_1.ToDevice(dev_1);
  net_2.ToDevice(dev_2);

  train_x_1.ToDevice(dev_1);
  train_y_1.ToDevice(dev_1);
  test_x.ToDevice(dev_1);
  test_y.ToDevice(dev_1);
  train_x_2.ToDevice(dev_2);
  train_y_2.ToDevice(dev_2);

  LOG(INFO) << "Launching thread...";
  std::thread t1 =
      net_1.TrainThread(50, num_epoch, train_x_1, train_y_1, test_x, test_y);
  std::thread t2 = net_2.TrainThread(50, num_epoch, train_x_2, train_y_2);
  t1.join();
  t2.join();
}
}  // namespace singa

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
