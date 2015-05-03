// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-15 21:54
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "proto/model.pb.h"
#include "utils/proto_helper.h"
namespace lapis {

// use const Message& m=..., otherwise may lead to segment fault
TEST(ProtoTest, ReadFromFile) {
  ModelProto model;
  LOG(INFO)<<"start....";
  lapis::ReadProtoFromTextFile("src/test/data/model.conf", &model);
  LOG(INFO)<<"after reading file...";
  EXPECT_STREQ("caffe_config", model.name().c_str());

  // layer and edge size
  const NetProto& net = model.net();
  EXPECT_EQ(15, net.layer().size());
  EXPECT_EQ(14, net.edge().size());
  LOG(INFO)<<"after size check...";

  // layer config
  LayerProto layer1 = net.layer().Get(1);
  EXPECT_STREQ("input_img", layer1.name().c_str());
  EXPECT_STREQ("DataLayer", layer1.type().c_str());
  LOG(INFO)<<"after datalayer check...";
  // edge config
  EdgeProto edge0 = net.edge().Get(0);
  EXPECT_STREQ("input_img-hidden1_relu", edge0.name().c_str());
  EXPECT_STREQ("ConvEdge", edge0.type().c_str());
  EXPECT_EQ(2, edge0.param().size());
  LOG(INFO)<<"after first edge check...";
  // param config
  ParamProto param1 = edge0.param().Get(0);
  EXPECT_TRUE(ParamProto::kGaussain == param1.init_method());
  EXPECT_EQ(0.0f, param1.mean());
  EXPECT_EQ(0.01f, param1.std());
  EXPECT_EQ(1.0f, param1.learning_rate_multiplier());
  LOG(INFO)<<"after param of first edge check...";

  ParamProto param2 = edge0.param().Get(1);
  EXPECT_TRUE(ParamProto::kConstant == param2.init_method());
  EXPECT_EQ(0.0f, param2.value());
  EXPECT_EQ(0.0f, param2.weight_decay_multiplier());
  LOG(INFO)<<"after param of second edge check...";

  // trainer config
  const TrainerProto& trainer = model.trainer();
  const SGDProto& sgd=trainer.sgd();
  EXPECT_EQ(227, sgd.train_batchsize());
  EXPECT_EQ(0.01f, sgd.base_learning_rate());
  EXPECT_TRUE(SGDProto::kStep== sgd.learning_rate_change());
  LOG(INFO)<<"after sgd check...";

  // data source config
  EXPECT_EQ(2,trainer.train_data().size());
  LOG(INFO)<<"after size check...";
  const DataSourceProto& data=trainer.train_data(0);
  LOG(INFO)<<"after get data...";
  EXPECT_STREQ("RGBDirSource", data.type().c_str());
  LOG(INFO)<<"after type check...";
  EXPECT_EQ(50000, data.size());
  EXPECT_EQ(3, data.channels());
  LOG(INFO)<<"after data source check...";
}
} // namespace lapis
