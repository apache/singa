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
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "singa/neuralnet/input_layer.h"
#include "singa/proto/job.pb.h"
#include "singa/proto/common.pb.h"

class RecordInputLayerTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    std::string path ="src/test/test.bin";
    auto* store = singa::io::CreateStore("kvfile");
    store->Open(path, singa::io::kCreate);
    {
    singa::RecordProto image;
    image.add_data(3.2);
    image.add_data(1);
    image.add_data(14.1);
    image.set_label(12);
    std::string val;
    image.SerializeToString(&val);
    store->Write("0", val);
    }

    {
    singa::SingleLabelImageRecord image;
    image.add_data(0.2);
    image.add_data(0);
    image.add_data(1.1);
    image.set_label(2);
    std::string val;
    image.SerializeToString(&val);
    store->Write("1", val);
    }

    {
    singa::SingleLabelImageRecord image;
    image.add_data(2.2);
    image.add_data(1);
    image.add_data(4.1);
    image.set_label(1);
    std::string val;
    image.SerializeToString(&val);
    store->Write("2", val);
    }
    store->Flush();
    store->Close();

    auto conf = image_conf.mutable_store_conf();
    conf->set_path(path);
    conf->set_batchsize(2);
    conf->add_shape(3);
    conf->set_backend("kvfile");
  }
  singa::LayerProto image_conf;
};

TEST_F(RecordInputLayerTest, Setup) {
  singa::RecordInputLayer layer;
  layer.Setup(image_conf, std::vector<singa::Layer*>{});
  EXPECT_EQ(2, static_cast<int>(layer.aux_data().size()));
  EXPECT_EQ(6, layer.data(nullptr).count());
}

TEST_F(RecordInputLayerTest, ComputeFeature) {
  singa::RecordInputLayer image;
  image.Setup(image_conf, std::vector<singa::Layer*>{});
  image.ComputeFeature(singa::kTrain, std::vector<singa::Layer*>{});

  EXPECT_EQ(12, image.aux_data()[0]);
  EXPECT_EQ(2, image.aux_data()[1]);
  auto data = image.data(nullptr);
  EXPECT_EQ(3.2f, data.cpu_data()[0]);
  EXPECT_EQ(14.1f, data.cpu_data()[2]);
  EXPECT_EQ(0.2f, data.cpu_data()[3]);
  EXPECT_EQ(1.1f, data.cpu_data()[5]);
}
TEST_F(RecordInputLayerTest, ComputeFeatureDeploy) {
  singa::RecordInputLayer image;
  image.Setup(image_conf, std::vector<singa::Layer*>{});
  image.ComputeFeature(singa::kDeploy, std::vector<singa::Layer*>{});

  auto data = image.data(nullptr);
  EXPECT_EQ(3.2f, data.cpu_data()[0]);
  EXPECT_EQ(14.1f, data.cpu_data()[2]);
  EXPECT_EQ(0.2f, data.cpu_data()[3]);
  EXPECT_EQ(1.1f, data.cpu_data()[5]);
}

TEST_F(RecordInputLayerTest, SeekToFirst) {
  singa::RecordInputLayer image;
  image.Setup(image_conf, std::vector<singa::Layer*>{});
  image.ComputeFeature(singa::kTrain, std::vector<singa::Layer*>{});
  image.ComputeFeature(singa::kTrain, std::vector<singa::Layer*>{});

  auto data = image.data(nullptr);
  EXPECT_EQ(2.2f, data.cpu_data()[0]);
  EXPECT_EQ(4.1f, data.cpu_data()[2]);
  EXPECT_EQ(3.2f, data.cpu_data()[3]);
  EXPECT_EQ(14.1f, data.cpu_data()[5]);
}
