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

// currently supports 'cudnn' and 'singacpp'
const std::string engine = "cudnn";
LayerConf GenConvConf(string name, int nb_filter, int kernel, int stride,
                      int pad, float std, float bias = .0f) {
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

LayerConf GenDenseConf(string name, int num_output, float std, float wd,
                       float bias = .0f) {
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
  auto bfill = bspec->mutable_filler();
  bfill->set_value(bias);

  return conf;
}

LayerConf GenLRNConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type(engine + "_lrn");
  LRNConf *lrn = conf.mutable_lrn_conf();
  lrn->set_local_size(5);
  lrn->set_alpha(1e-04);
  lrn->set_beta(0.75);
  return conf;
}

LayerConf GenFlattenConf(string name) {
  LayerConf conf;
  conf.set_name(name);
  conf.set_type("singa_flatten");
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

FeedForwardNet CreateNet() {
  FeedForwardNet net;
  Shape s{3, 227, 227};

  net.Add(GenConvConf("conv1", 96, 11, 4, 0, 0.01), &s);
  net.Add(GenReLUConf("relu1"));
  net.Add(GenPoolingConf("pool1", true, 3, 2, 0));
  net.Add(GenLRNConf("lrn1"));
  net.Add(GenConvConf("conv2", 256, 5, 1, 2, 0.01, 1.0));
  net.Add(GenReLUConf("relu2"));
  net.Add(GenPoolingConf("pool2", true, 3, 2, 0));
  net.Add(GenLRNConf("lrn2"));
  net.Add(GenConvConf("conv3", 384, 3, 1, 1, 0.01));
  net.Add(GenReLUConf("relu3"));
  net.Add(GenConvConf("conv4", 384, 3, 1, 1, 0.01, 1.0));
  net.Add(GenReLUConf("relu4"));
  net.Add(GenConvConf("conv5", 256, 3, 1, 1, 0.01, 1.0));
  net.Add(GenReLUConf("relu5"));
  net.Add(GenPoolingConf("pool5", true, 3, 2, 0));
  net.Add(GenFlattenConf("flat"));
  net.Add(GenDenseConf("ip6", 4096, 0.005, 1, 1.0));
  net.Add(GenReLUConf("relu6"));
  net.Add(GenDropoutConf("drop6", 0.5));
  net.Add(GenDenseConf("ip7", 4096, 0.005, 1, 1.0));
  net.Add(GenReLUConf("relu7"));
  net.Add(GenDropoutConf("drop7", 0.5));
  net.Add(GenDenseConf("ip8", 1000, 0.01, 1));

  return net;
}

void TrainOneEpoch(FeedForwardNet &net, ILSVRC &data,
                   std::shared_ptr<Device> device, int epoch, string bin_folder,
                   size_t num_train_files, size_t batchsize, float lr,
                   Channel *train_ch, size_t pfreq, int nthreads) {
  float loss = 0.0f, metric = 0.0f;
  float load_time = 0.0f, train_time = 0.0f;
  size_t b = 0;
  size_t n_read;
  Timer timer, ttr;
  Tensor prefetch_x(Shape{batchsize, 3, kCropSize, kCropSize}),
      prefetch_y(Shape{batchsize}, kInt);
  string binfile = bin_folder + "/train1.bin";
  timer.Tick();
  data.LoadData(kTrain, binfile, batchsize, &prefetch_x, &prefetch_y, &n_read,
                nthreads);
  load_time += timer.Elapsed();
  CHECK_EQ(n_read, batchsize);
  Tensor train_x(prefetch_x.shape(), device);
  Tensor train_y(prefetch_y.shape(), device, kInt);
  std::thread th;
  for (size_t fno = 1; fno <= num_train_files; fno++) {
    binfile = bin_folder + "/train" + std::to_string(fno) + ".bin";
    while (true) {
      if (th.joinable()) {
        th.join();
        load_time += timer.Elapsed();
        // LOG(INFO) << "num of samples: " << n_read;
        if (n_read < batchsize) {
          if (n_read > 0) {
            LOG(WARNING) << "Pls set batchsize to make num_total_samples "
                         << "% batchsize == 0. Otherwise, the last " << n_read
                         << " samples would not be used";
          }
          break;
        }
      }
      if (n_read == batchsize) {
        train_x.CopyData(prefetch_x);
        train_y.CopyData(prefetch_y);
      }
      timer.Tick();
      th = data.AsyncLoadData(kTrain, binfile, batchsize, &prefetch_x,
                              &prefetch_y, &n_read, nthreads);
      if (n_read < batchsize) continue;
      CHECK_EQ(train_x.shape(0), train_y.shape(0));
      ttr.Tick();
      auto ret = net.TrainOnBatch(epoch, train_x, train_y);
      train_time += ttr.Elapsed();
      loss += ret.first;
      metric += ret.second;
      b++;
    }
    if (b % pfreq == 0) {
      train_ch->Send(
          "Epoch " + std::to_string(epoch) +
          ", training loss = " + std::to_string(loss / b) + ", accuracy = " +
          std::to_string(metric / b) + ", lr = " + std::to_string(lr) +
          ", time of loading " + std::to_string(batchsize) +
          " images = " + std::to_string(load_time / b) +
          " ms, time of training (batchsize = " + std::to_string(batchsize) +
          ") = " + std::to_string(train_time / b) + " ms.");
      loss = 0.0f;
      metric = 0.0f;
      load_time = 0.0f;
      train_time = 0.0f;
      b = 0;
    }
  }
}

void TestOneEpoch(FeedForwardNet &net, ILSVRC &data,
                  std::shared_ptr<Device> device, int epoch, string bin_folder,
                  size_t num_test_images, size_t batchsize, Channel *val_ch,
                  int nthreads) {
  float loss = 0.0f, metric = 0.0f;
  float load_time = 0.0f, eval_time = 0.0f;
  size_t n_read;
  string binfile = bin_folder + "/test.bin";
  Timer timer, tte;
  Tensor prefetch_x, prefetch_y;
  timer.Tick();
  data.LoadData(kEval, binfile, batchsize, &prefetch_x, &prefetch_y, &n_read,
                nthreads);
  load_time += timer.Elapsed();
  Tensor test_x(prefetch_x.shape(), device);
  Tensor test_y(prefetch_y.shape(), device, kInt);
  int remain = (int)num_test_images - n_read;
  CHECK_EQ(n_read, batchsize);
  std::thread th;
  while (true) {
    if (th.joinable()) {
      th.join();
      load_time += timer.Elapsed();
      remain -= n_read;
      if (remain < 0) break;
      if (n_read < batchsize) break;
    }
    test_x.CopyData(prefetch_x);
    test_y.CopyData(prefetch_y);
    timer.Tick();
    th = data.AsyncLoadData(kEval, binfile, batchsize, &prefetch_x, &prefetch_y,
                            &n_read, nthreads);

    CHECK_EQ(test_x.shape(0), test_y.shape(0));
    tte.Tick();
    auto ret = net.EvaluateOnBatch(test_x, test_y);
    eval_time += tte.Elapsed();
    ret.first.ToHost();
    ret.second.ToHost();
    loss += Sum(ret.first);
    metric += Sum(ret.second);
  }
  loss /= num_test_images;
  metric /= num_test_images;
  val_ch->Send("Epoch " + std::to_string(epoch) + ", val loss = " +
               std::to_string(loss) + ", accuracy = " + std::to_string(metric) +
               ", time of loading " + std::to_string(num_test_images) +
               " images = " + std::to_string(load_time) +
               " ms, time of evaluating " + std::to_string(num_test_images) +
               " images = " + std::to_string(eval_time) + " ms.");
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
           string bin_folder, size_t num_train_images, size_t num_test_images,
           size_t pfreq, int nthreads) {
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
  size_t num_train_files = num_train_images / train_file_size +
                           (num_train_images % train_file_size ? 1 : 0);
  for (int epoch = 0; epoch < num_epoch; epoch++) {
    float epoch_lr = sgd.GetLearningRate(epoch);
    TrainOneEpoch(net, data, cuda, epoch, bin_folder, num_train_files,
                  batchsize, epoch_lr, train_ch, pfreq, nthreads);
    if (epoch % 10 == 0 && epoch > 0) {
      string prefix = "snapshot_epoch" + std::to_string(epoch);
      Checkpoint(net, prefix);
    }
    TestOneEpoch(net, data, cuda, epoch, bin_folder, num_test_images, batchsize,
                 val_ch, nthreads);
  }
}
}  // namespace singa

int main(int argc, char **argv) {
  singa::InitChannel(nullptr);
  int pos = singa::ArgPos(argc, argv, "-h");
  if (pos != -1) {
    std::cout
        << "Usage:\n"
        << "\t-epoch <int>: number of epoch to be trained, default is 90;\n"
        << "\t-lr <float>: base learning rate;\n"
        << "\t-batchsize <int>: batchsize, it should be changed regarding "
           "to your memory;\n"
        << "\t-filesize <int>: number of training images that stores in "
           "each binary file;\n"
        << "\t-ntrain <int>: number of training images;\n"
        << "\t-ntest <int>: number of test images;\n"
        << "\t-data <folder>: the folder which stores the binary files;\n"
        << "\t-pfreq <int>: the frequency(in batch) of printing current "
           "model status(loss and accuracy);\n"
        << "\t-nthreads <int>`: the number of threads to load data which "
           "feed to the model.\n";
    return 0;
  }
  pos = singa::ArgPos(argc, argv, "-epoch");
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
  string bin_folder = "imagenet_data";
  if (pos != -1) bin_folder = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-pfreq");
  size_t pfreq = 100;
  if (pos != -1) pfreq = atoi(argv[pos + 1]);

  pos = singa::ArgPos(argc, argv, "-nthreads");
  int nthreads = 12;
  if (pos != -1) nthreads = atoi(argv[pos + 1]);

  LOG(INFO) << "Start training";
  singa::Train(nEpoch, lr, batchsize, train_file_size, bin_folder,
               num_train_images, num_test_images, pfreq, nthreads);
  LOG(INFO) << "End training";
}
#endif
