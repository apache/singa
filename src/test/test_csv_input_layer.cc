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
#include <fstream>

#include "gtest/gtest.h"
#include "singa/neuralnet/input_layer.h"
#include "singa/proto/job.pb.h"

class CSVInputLayerTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    std::string path ="src/test/test.csv";
    std::ofstream ofs(path, std::ofstream::out);
    ASSERT_TRUE(ofs.is_open());
    ofs << "12,3.2,1,14.1\n";
    ofs << "2,0.2,0,1.1\n";
    ofs << "1,2.2,1,4.1\n";
    ofs.close();
    auto conf = csv_conf.mutable_store_conf();
    conf->set_path(path);
    conf->set_batchsize(2);
    conf->add_shape(3);
    conf->set_backend("textfile");
  }
  singa::LayerProto csv_conf;
};

TEST_F(CSVInputLayerTest, Setup) {
  singa::CSVInputLayer layer;
  layer.Setup(csv_conf, std::vector<singa::Layer*>{});
  EXPECT_EQ(2, static_cast<int>(layer.aux_data().size()));
  EXPECT_EQ(6, layer.data(nullptr).count());
}

TEST_F(CSVInputLayerTest, ComputeFeature) {
  singa::CSVInputLayer csv;
  csv.Setup(csv_conf, std::vector<singa::Layer*>{});
  csv.ComputeFeature(singa::kTrain, std::vector<singa::Layer*>{});

  EXPECT_EQ(12, csv.aux_data()[0]);
  EXPECT_EQ(2, csv.aux_data()[1]);
  auto data = csv.data(nullptr);
  EXPECT_EQ(3.2f, data.cpu_data()[0]);
  EXPECT_EQ(14.1f, data.cpu_data()[2]);
  EXPECT_EQ(0.2f, data.cpu_data()[3]);
  EXPECT_EQ(1.1f, data.cpu_data()[5]);
}
TEST_F(CSVInputLayerTest, ComputeFeatureDeploy) {
  singa::CSVInputLayer csv;
  csv_conf.mutable_store_conf()->set_shape(0, 4);
  csv.Setup(csv_conf, std::vector<singa::Layer*>{});
  csv.ComputeFeature(singa::kDeploy, std::vector<singa::Layer*>{});

  auto data = csv.data(nullptr);
  EXPECT_EQ(12.f, data.cpu_data()[0]);
  EXPECT_EQ(1.f, data.cpu_data()[2]);
  EXPECT_EQ(14.1f, data.cpu_data()[3]);
  EXPECT_EQ(0.2f, data.cpu_data()[5]);
}

TEST_F(CSVInputLayerTest, SeekToFirst) {
  singa::CSVInputLayer csv;
  csv.Setup(csv_conf, std::vector<singa::Layer*>{});
  csv.ComputeFeature(singa::kTrain, std::vector<singa::Layer*>{});
  csv.ComputeFeature(singa::kTrain, std::vector<singa::Layer*>{});

  auto data = csv.data(nullptr);
  EXPECT_EQ(2.2f, data.cpu_data()[0]);
  EXPECT_EQ(4.1f, data.cpu_data()[2]);
  EXPECT_EQ(3.2f, data.cpu_data()[3]);
  EXPECT_EQ(14.1f, data.cpu_data()[5]);
}
