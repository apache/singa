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

#include "singa/singa_config.h"
#ifdef USE_OPENCV
#include <cmath>
#include "../../src/model/layer/cudnn_activation.h"
#include "../../src/model/layer/cudnn_convolution.h"
#include "../../src/model/layer/cudnn_dropout.h"
#include "../../src/model/layer/cudnn_lrn.h"
#include "../../src/model/layer/cudnn_pooling.h"
#include "../../src/model/layer/dense.h"
#include "../../src/model/layer/flatten.h"
#include "./ilsvrc12.h"
#include "singa/io/snapshot.h"
#include "singa/model/feed_forward_net.h"
#include "singa/model/initializer.h"
#include "singa/model/metric.h"
#include "singa/model/optimizer.h"
#include "singa/utils/channel.h"
#include "singa/utils/string.h"
#include "singa/utils/timer.h"
namespace singa {

LayerConf GenConvConf(string name, int nb_filter, int kernel, int stride,
                      int pad, float std, float bias = .0f) {
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
  bspec->set_decay_mult(0);
  auto bfill = bspec->mutable_filler();
  bfill->set_value(bias);
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

LayerConf GenDenseConf(string name, int num_output, float std, float wd,
                       float bias = .0f) {
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
  auto bfill = bspec->mutable_filler();
  bfill->set_value(bias);

  return conf;
}

LayerConf GenLRNConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("CudnnLRN");
  LRNConf *lrn = conf.mutable_lrn_conf();
  lrn->set_local_size(5);
  lrn->set_alpha(1e-04);
  lrn->set_beta(0.75);
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

FeedForwardNet CreateNet() {
  FeedForwardNet net;
  Shape s{3, 227, 227};

  net.Add(new CudnnConvolution(), GenConvConf("conv1", 96, 11, 4, 0, 0.01), &s);
  net.Add(new CudnnActivation(), GenReLUConf("relu1"));
  net.Add(new CudnnPooling(), GenPoolingConf("pool1", true, 3, 2, 0));
  net.Add(new CudnnLRN(), GenLRNConf("lrn1"));
  net.Add(new CudnnConvolution(),
          GenConvConf("conv2", 256, 5, 1, 2, 0.01, 1.0));
  net.Add(new CudnnActivation(), GenReLUConf("relu2"));
  net.Add(new CudnnPooling(), GenPoolingConf("pool2", true, 3, 2, 0));
  net.Add(new CudnnLRN(), GenLRNConf("lrn2"));
  net.Add(new CudnnConvolution(), GenConvConf("conv3", 384, 3, 1, 1, 0.01));
  net.Add(new CudnnActivation(), GenReLUConf("relu3"));
  net.Add(new CudnnConvolution(),
          GenConvConf("conv4", 384, 3, 1, 1, 0.01, 1.0));
  net.Add(new CudnnActivation(), GenReLUConf("relu4"));
  net.Add(new CudnnConvolution(),
          GenConvConf("conv5", 256, 3, 1, 1, 0.01, 1.0));
  net.Add(new CudnnActivation(), GenReLUConf("relu5"));
  net.Add(new CudnnPooling(), GenPoolingConf("pool5", true, 3, 2, 0));
  net.Add(new Flatten(), GenFlattenConf("flat"));
  net.Add(new Dense(), GenDenseConf("ip6", 4096, 0.005, 1, 1.0));
  net.Add(new CudnnActivation(), GenReLUConf("relu6"));
  net.Add(new CudnnDropout(), GenDropoutConf("drop6", 0.5));
  net.Add(new Dense(), GenDenseConf("ip7", 4096, 0.005, 1, 1.0));
  net.Add(new CudnnActivation(), GenReLUConf("relu7"));
  net.Add(new CudnnDropout(), GenDropoutConf("drop7", 0.5));
  net.Add(new Dense(), GenDenseConf("ip8", 1000, 0.01, 1));

  return net;
}

void TrainOneEpoch(FeedForwardNet &net, ILSVRC &data,
                                      std::shared_ptr<Device> device, int epoch,
                                      string bin_folder,
                                      size_t num_train_images,
                                      size_t train_file_size,
                                      size_t read_size,
                                      float lr,
                                      Channel *train_ch) {
  size_t num_train_files = num_train_images / train_file_size +
                           (num_train_images % train_file_size ? 1 : 0);
  string mean_path = bin_folder + "/mean.bin";
  float loss = 0.0f, metric = 0.0f;
  float load_time = 0.0f, train_time = 0.0f;
  size_t b = 0;
  size_t n_read;
  Timer timer, ttr;
  Tensor prefetch_x, prefetch_y;
  //prefetch_x.ToDevice(device);
  //prefetch_y.ToDevice(device);
  string binfile = bin_folder + "/train1.bin";
  timer.Tick();
  data.LoadData(kTrain, binfile, read_size, &prefetch_x, &prefetch_y, &n_read);
  load_time += timer.Elapsed();
  CHECK_EQ(n_read, read_size);
  Tensor train_x(prefetch_x.shape(), device);
  Tensor train_y(prefetch_y.shape(), device, kInt);
  std::thread th;
  //LOG(INFO) << "Total num of training files: " << num_train_files;
  for (size_t fno = 1; fno <= num_train_files; fno++) {
    binfile = bin_folder + "/train" + std::to_string(fno) + ".bin";
    //LOG(INFO) << "Load data from " << binfile;
    while (true) {
      if (th.joinable()) {
        th.join();
        load_time += timer.Elapsed();
        //LOG(INFO) << "num of samples: " << n_read;
        if (n_read < read_size) {
          if (n_read > 0) {
            LOG(WARNING) << "Pls set batchsize to make num_total_samples "
              << "% batchsize == 0. Otherwise, the last " << n_read
              << " samples would not be used";
          }
          break;
        }
      }
      if (n_read == read_size) {
        train_x.CopyData(prefetch_x);
        train_y.CopyData(prefetch_y);
      }
      //LOG(INFO) << "train_x.L1(): " << train_x.L1();
      timer.Tick();
      th = data.AsyncLoadData(kTrain, binfile, read_size, &prefetch_x, &prefetch_y, &n_read);
      if (n_read < read_size) continue;
      CHECK_EQ(train_x.shape(0), train_y.shape(0));
 //     train_x.ToDevice(device);
//      train_y.ToDevice(device);
      ttr.Tick();
      auto ret = net.TrainOnBatch(epoch, train_x, train_y);
      train_time += ttr.Elapsed();
      loss += ret.first;
      metric += ret.second;
      b++;
      //LOG(INFO) << "batch " << b << ", loss: " << ret.first;
    }
    if (fno % 20 == 0) {
      //LOG(INFO) << "num of batch: " << b;
      train_ch->Send("Epoch " + std::to_string(epoch) + ", training loss = " +
                      std::to_string(loss / b) + ", accuracy = " +
                      std::to_string(metric / b) + ", lr = " +
                      std::to_string(lr)
                      + ", time of loading " + std::to_string(read_size) + " images = "
                      + std::to_string(load_time / b) + " ms, time of training (batchsize = "
                      + std::to_string(read_size) + ") = " + std::to_string(train_time / b) + " ms.");
      loss = 0.0f;
      metric = 0.0f;
      load_time = 0.0f;
      train_time = 0.0f;
      b = 0;
    }
  }
}

void TestOneEpoch(FeedForwardNet &net, ILSVRC &data,
                                     std::shared_ptr<Device> device, int epoch,
                                     string bin_folder, size_t num_test_images,
                                     size_t read_size,
                                     Channel *val_ch) {
  float loss = 0.0f, metric = 0.0f;
  float load_time = 0.0f, eval_time = 0.0f;
  size_t n_read;
  string binfile = bin_folder + "/test.bin";
  Timer timer, tte;
  Tensor prefetch_x, prefetch_y;
  //prefetch_x.ToDevice(device);
  //prefetch_y.ToDevice(device);
  timer.Tick();
  data.LoadData(kEval, binfile, read_size, &prefetch_x, &prefetch_y, &n_read);
  load_time += timer.Elapsed();
  Tensor test_x(prefetch_x.shape(), device);
  Tensor test_y(prefetch_y.shape(), device, kInt);
  int remain = (int)num_test_images - n_read;
  //LOG(INFO) << "num of test images: " << remain;
  CHECK_EQ(n_read, read_size);
  //test_x.ResetLike(prefetch_x);
  //test_y.ResetLike(prefetch_y);
  std::thread th;
  while (true) {
    if (th.joinable()) {
      th.join();
      load_time += timer.Elapsed();
      remain -= n_read;
      if (remain < 0) break;
      if (n_read < read_size) break;
    }
    //LOG(INFO) << "num of test images: " << remain;
    test_x.CopyData(prefetch_x);
    test_y.CopyData(prefetch_y);
    timer.Tick();
    th = data.AsyncLoadData(kEval, binfile, read_size, &prefetch_x, &prefetch_y, &n_read);

    CHECK_EQ(test_x.shape(0), test_y.shape(0));
    //test_x.ToDevice(device);
    //test_y.ToDevice(device);
    tte.Tick();
    auto ret = net.EvaluateOnBatch(test_x, test_y);
    eval_time += tte.Elapsed();
    ret.first.ToHost();
    ret.second.ToHost();
    loss += Sum(ret.first);
    metric += Sum(ret.second);
    //LOG(INFO) << "loss: " << loss;
    //LOG(INFO) << "metric: " << metric;
  }
  loss /= num_test_images;
  metric /= num_test_images;
  val_ch->Send("Epoch " + std::to_string(epoch) + ", val loss = " +
                std::to_string(loss) + ", accuracy = " +
                std::to_string(metric)
                + ", time of loading " + std::to_string(num_test_images) + " images = "
                + std::to_string(load_time) + " ms, time of evaluating "
                + std::to_string(num_test_images) + " images = " + std::to_string(eval_time) + " ms.");
}

void Checkpoint(FeedForwardNet &net, string prefix) {
  Snapshot snapshot(prefix, Snapshot::kWrite, 200);
  auto names = net.GetParamNames();
  auto values = net.GetParamValues();
  for (size_t k = 0; k < names.size(); k++) {
    values.at(k).ToHost();
    snapshot.Write(names.at(k), values.at(k));

  }
  LOG(INFO) << "Write snapshot into " << prefix;
}

void Train(int num_epoch, float lr, size_t batchsize, size_t train_file_size,
           string bin_folder, size_t num_train_images, size_t num_test_images) {
  ILSVRC data;
  data.ReadMean(bin_folder + "/mean.bin");
  auto net = CreateNet();
  auto cuda = std::make_shared<CudaGPU>(0);
  net.ToDevice(cuda);
  SGD sgd;
  OptimizerConf opt_conf;
  opt_conf.set_momentum(0.9);
  auto reg = opt_conf.mutable_regularizer();
  reg->set_coefficient(0.0005);
  sgd.Setup(opt_conf);
  sgd.SetLearningRateGenerator(
      [lr](int epoch) { return lr * std::pow(0.1, epoch / 20); });

  SoftmaxCrossEntropy loss;
  Accuracy acc;
  net.Compile(true, &sgd, &loss, &acc);

  Channel *train_ch = GetChannel("train_perf");
  train_ch->EnableDestStderr(true);
  Channel *val_ch = GetChannel("val_perf");
  val_ch->EnableDestStderr(true);
  for (int epoch = 0; epoch < num_epoch; epoch++) {
    float epoch_lr = sgd.GetLearningRate(epoch);
    TrainOneEpoch(net, data, cuda, epoch, bin_folder, num_train_images,
                  train_file_size, batchsize, epoch_lr, train_ch);
    if (epoch % 10 == 0 && epoch > 0) {
      string prefix = "snapshot_epoch" + std::to_string(epoch);
      Checkpoint(net, prefix);
    }
    TestOneEpoch(net, data, cuda, epoch, bin_folder, num_test_images, batchsize, val_ch);
  }
}
}

int main(int argc, char **argv) {
  singa::InitChannel(nullptr);
  int pos = singa::ArgPos(argc, argv, "-epoch");
  int nEpoch = 90;
  if (pos != -1) nEpoch = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-lr");
  float lr = 0.01;
  if (pos != -1) lr = atof(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-batchsize");
  int batchsize = 256;
  if (pos != -1) batchsize = atof(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-filesize");
  size_t train_file_size = 1280;
  if (pos != -1) train_file_size = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-ntrain");
  size_t num_train_images = 1281167;
  if (pos != -1) num_train_images = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-ntest");
  size_t num_test_images = 50000;
  if (pos != -1) num_test_images = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-data");
  string bin_folder = "/home/xiangrui/imagenet_data";
  if (pos != -1) bin_folder = argv[pos + 1];

  LOG(INFO) << "Start training";
  singa::Train(nEpoch, lr, batchsize, train_file_size, bin_folder,
               num_train_images, num_test_images);
  LOG(INFO) << "End training";
}
#endif
