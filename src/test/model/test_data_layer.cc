// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-08-01 16:09

#include <gtest/gtest.h>
#include <glog/logging.h>
#include <map>
#include <vector>

#include "model/data_layer.h"
#include "model/trainer.h"
#include "model/sgd_trainer.h"
#include "model/conv_edge.h"
#include "model/relu_layer.h"
#include "proto/model.pb.h"

#include "utils/proto_helper.h"

namespace lapis {
class ModelTest : public ::testing::Test {
 public:
  ModelTest () {
    ReadProtoFromTextFile("src/test/data/model.conf", &model_proto);
  }
 protected:
  ModelProto model_proto;
};
/**********************************************************************
 * DataLayer Test
 **********************************************************************/
class DataLayerTest : public ModelTest {
 public:
   DataLayerTest() {
     label_layer.Init(model_proto.net().layer(0));
     img_layer.Init(model_proto.net().layer(1));
     Trainer::InitDataSource(model_proto.trainer().train_data(), &sources);
     EXPECT_EQ(2, sources.size());
     sources[0]->LoadData(nullptr);
     sources[1]->LoadData(nullptr);
     DLOG(INFO)<<"after init datasources";
     label_layer.Setup(2, TrainerProto::kBackPropagation, sources);
     DLOG(INFO)<<"after setup label layer";
     img_layer.Setup(2, TrainerProto::kBackPropagation, sources);
     DLOG(INFO)<<"after setup img layer";
   }
   ~DataLayerTest() {
     for(auto& source: sources)
       delete source;
   }
 protected:
  DataLayer img_layer, label_layer;
  std::vector<DataSource*> sources;
};

TEST_F(DataLayerTest, InitSetupForward) {
  EXPECT_TRUE(label_layer.HasInput());
  EXPECT_TRUE(img_layer.HasInput());
  EXPECT_STREQ("DataLayer", DataLayer::kType.c_str());

  EXPECT_EQ(2, label_layer.feature(nullptr).num());
  EXPECT_EQ(1, label_layer.feature(nullptr).channels());
  EXPECT_EQ(1, label_layer.feature(nullptr).height());
  EXPECT_EQ(1, label_layer.feature(nullptr).width());

  EXPECT_EQ(2, img_layer.feature(nullptr).num());
  EXPECT_EQ(3, img_layer.feature(nullptr).channels());
  EXPECT_EQ(227, img_layer.feature(nullptr).height());
  EXPECT_EQ(227, img_layer.feature(nullptr).width());

  img_layer.Forward();
}
// TODO(wangwei) test this after outgoing edges are tested

/**********************************************************************
 * ConvEdge Test
 **********************************************************************/
class ConvEdgeTest : public DataLayerTest {
 public:
  ConvEdgeTest() {
    relu.Init(model_proto.net().layer(2));
    DLOG(INFO)<<"init both layers";
    layer_map["input_img"]=&img_layer;
    layer_map["hidden1_relu"]=&relu;

    edge_proto=model_proto.net().edge(0);
    convedge.Init(edge_proto, layer_map);
    convedge.Setup(true);
  }
 protected:
  std::map<std::string, Layer*> layer_map;
  ConvEdge convedge;
  EdgeProto edge_proto;
  ReLULayer relu;
};

TEST_F(ConvEdgeTest, InitSetupForward) {
  Layer* dest=layer_map.at("hidden1_relu");
  Blob &b=dest->feature(&convedge);
  EXPECT_EQ(0,b.num());
  convedge.SetupTopBlob(&b);
  int conv_height = (227 + 2 * edge_proto.pad() - edge_proto.kernel_size())
    / edge_proto.stride() + 1;
  int conv_width=conv_height;
  CHECK_EQ(2, b.num());
  CHECK_EQ(edge_proto.num_output(), b.channels());
  CHECK_EQ(conv_height, b.height());
  CHECK_EQ(conv_width, b.width());
  DLOG(INFO)<<"after shape check";

  Layer* src=layer_map["input_img"];
  convedge.Forward(src->feature(&convedge), &b, true);
}

/**********************************************************************
 * ReLULayer Test
 **********************************************************************/
class ReLULayerTest : public ConvEdgeTest {
 public:
  ReLULayerTest() {
    relu.Setup(2, TrainerProto::kBackPropagation, sources);
    relu_proto=model_proto.net().layer(3);
  }
 protected:
  LayerProto relu_proto;
};

TEST_F(ReLULayerTest, ForwardWithoutDropout) {
  EXPECT_EQ(2, relu.feature(&convedge).num());
  EXPECT_EQ(2, relu.gradient(&convedge).num());

  relu.Forward();
}
/**********************************************************************
 * PoolingEdge Test
class PoolingEdgeTest : public ReLULayerTest {
 public:
  PoolingEdgeTest() {
    linearlayer.Init(model.net().layer(3));
    pooledge.Init(model.net().edge(1));
  }

 protected:
  PoolingEdge pooledge;
  LinearLayer linearlayer;
}
 **********************************************************************/
/**********************************************************************
 * LinearLayer Test
 **********************************************************************/

/**********************************************************************
 * LRNEdge Test
 **********************************************************************/

/**********************************************************************
 * InnerProductEdge Test
 **********************************************************************/

/**********************************************************************
 * SoftmaxLayerLossEdge Test
 **********************************************************************/




/**********************************************************************
 * SGDTrainer Test
 **********************************************************************/
class SGDTrainerTest : public ModelTest {
 protected:
  SGDTrainer sgd;
};

TEST_F(SGDTrainerTest, Init) {
  sgd.Init(model_proto.trainer());
  EXPECT_TRUE(Trainer::phase==Phase::kInit);
}

}  // namespace lapis
