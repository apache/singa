#include <gtest/gtest.h>
#include <model/neuralnet.h>
#include "proto/model.pb.h"
#include "utils/common.h"
#include "utils/param_updater.h"

using namespace singa;
NetProto CreateMLPProto(){
  ModelProto model;
  ReadProtoFromTextFile("examples/mnist/mlp.conf", &model);
  return model.neuralnet();
}
TEST(NeuralnetTest, BP){
  ModelProto model;
  ReadProtoFromTextFile("examples/mnist/mlp.conf", &model);

  AdaGradUpdater updater;
  updater.Init(model.solver().updater());

  NeuralNet net(model.neuralnet());
  auto layers=net.layers();
  for(int i=0;i<3;i++){
    bool firstlayer=true;
    for(auto& layer: layers){
      layer->ComputeFeature();
      if(firstlayer){
        DataLayer* dl=static_cast<DataLayer*>(layer.get());
        dl->CompletePrefetch();
        firstlayer=false;
      }
    }

    for(int k=layers.size()-1;k>=0;k--){
      layers[k]->ComputeGradient();
      for(Param* param: layers[k]->GetParams())
        updater.Update(i, param);
    }
  }
}
NetProto CreateConvNetProto(){
  NetProto proto;
  LayerProto *layer;

  layer=proto.add_layer();
  layer->set_name("data");
  layer->set_type("kShardData");
  DataProto *data=layer->mutable_data_param();
  data->set_batchsize(8);
  data->set_path("/data1/wangwei/singa/data/mnist/train/");

  // 4x3x10x10
  layer=proto.add_layer();
  layer->set_name("mnist");
  layer->set_type("kMnistImage");
  layer->add_srclayers("data");

  // 4x1
  layer=proto.add_layer();
  layer->set_name("label");
  layer->set_type("kLabel");
  layer->add_srclayers("data");

  // 4x8x9x9
  layer=proto.add_layer();
  layer->set_name("conv1");
  layer->set_type("kConvolution");
  layer->add_srclayers("mnist");
  layer->add_param();
  layer->add_param();
  ConvolutionProto *conv=layer->mutable_convolution_param();
  conv->set_num_filters(8);
  conv->set_kernel(2);

  // 4x8x9x9
  layer=proto.add_layer();
  layer->set_name("relu1");
  layer->set_type("kReLU");
  layer->add_srclayers("conv1");

  // 4x8x4x4
  layer=proto.add_layer();
  layer->set_name("pool1");
  layer->set_type("kPooling");
  layer->add_srclayers("relu1");
  PoolingProto *pool=layer->mutable_pooling_param();
  pool->set_kernel(4);
  pool->set_stride(2);

  // 4x10
  layer=proto.add_layer();
  layer->set_name("fc1");
  layer->set_type("kInnerProduct");
  layer->add_srclayers("pool1");
  layer->add_param();
  layer->add_param();
  InnerProductProto *inner=layer->mutable_inner_product_param();
  inner->set_num_output(10);

  // 4x10
  layer=proto.add_layer();
  layer->set_name("loss");
  layer->set_type("kSoftmaxLoss");
  layer->add_srclayers("fc1");
  layer->add_srclayers("label");

  return proto;
}

TEST(NeuralNetTest, NoPartition){
  NetProto proto=CreateConvNetProto();
  NeuralNet net(proto);
  const auto& layers=net.layers();
  ASSERT_EQ(8, layers.size());
  ASSERT_EQ("data", layers.at(0)->name());
  ASSERT_EQ("loss", layers.at(7)->name());
}

TEST(NeuralNetTest, DataPartition){
  NetProto proto=CreateConvNetProto();
  proto.set_partition_type(kDataPartition);
  NeuralNet net(proto, 3);
  const auto& layers=net.layers();
  ASSERT_EQ(28, layers.size());
  ASSERT_EQ("data", layers.at(0)->name());
}
TEST(NeuralNetTest, LayerPartition){
  NetProto proto=CreateConvNetProto();
  proto.set_partition_type(kLayerPartition);
  NeuralNet net(proto, 2);
 // const auto& layers=net.layers();
}
TEST(NeuralNetTest, HyridPartition){
  NetProto proto=CreateConvNetProto();
  int num_layers=proto.layer_size();
  proto.mutable_layer(num_layers-2)->set_partition_type(kDataPartition);
  proto.mutable_layer(num_layers-1)->set_partition_type(kDataPartition);
  proto.set_partition_type(kLayerPartition);
  NeuralNet net(proto, 2);
}


