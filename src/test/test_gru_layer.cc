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
#include <iostream>
using namespace std;


#include "gtest/gtest.h"
#include "singa/neuralnet/neuron_layer/gru.h"
#include "singa/neuralnet/input_layer/csv.h"
#include "singa/driver.h"
#include "singa/proto/job.pb.h"

using namespace singa;

class GRULayerTest: public ::testing::Test {
protected:
	virtual void SetUp() {
		// Initialize the settings for the first input-layer
		std::string path1 = "src/test/gru-in-1.csv"; // path of a csv file
		std::ofstream ofs1(path1, std::ofstream::out);
		ASSERT_TRUE(ofs1.is_open());
		ofs1 << "0,0,0,1\n";
		ofs1 << "0,0,1,0\n";
		ofs1.close();
		auto conf1 = in1_conf.mutable_store_conf();
		conf1->set_path(path1);
		conf1->set_batchsize(2);
		conf1->add_shape(4);
		conf1->set_backend("textfile");

		// Initialize the settings for the second input-layer
		std::string path2 = "src/test/gru-in-2.csv"; // path of a csv file
		std::ofstream ofs2(path2, std::ofstream::out);
		ASSERT_TRUE(ofs2.is_open());
		ofs2 << "0,1,0,0\n";
		ofs2 << "1,0,0,0\n";
		ofs2.close();
		auto conf2 = in2_conf.mutable_store_conf();
		conf2->set_path(path2);

		conf2->set_batchsize(2);
		conf2->add_shape(4);
		conf2->set_backend("textfile");

		gru1_conf.mutable_gru_conf() -> set_dim_hidden(2);
		gru1_conf.mutable_gru_conf() -> set_bias_term(true);
		for (int i = 0; i < 9; i ++) {
			gru1_conf.add_param();
		}


		gru1_conf.mutable_param(0)->set_name("wzhx1");
		gru1_conf.mutable_param(0)->set_type(kParam);
		gru1_conf.mutable_param(0)->mutable_init()->set_type(kConstant);
		gru1_conf.mutable_param(0)->mutable_init()->set_value(0.5f);

		gru1_conf.mutable_param(1)->set_name("wzhh1");
		gru1_conf.mutable_param(1)->set_type(kParam);
		gru1_conf.mutable_param(1)->mutable_init()->set_type(kUniform);
		gru1_conf.mutable_param(2)->set_name("bz1");
		gru1_conf.mutable_param(2)->set_type(kParam);
		gru1_conf.mutable_param(2)->mutable_init()->set_type(kUniform);
		gru1_conf.mutable_param(3)->set_name("wrhx1");
		gru1_conf.mutable_param(3)->set_type(kParam);
		gru1_conf.mutable_param(3)->mutable_init()->set_type(kUniform);
		gru1_conf.mutable_param(4)->set_name("wrhh1");
		gru1_conf.mutable_param(4)->set_type(kParam);
		gru1_conf.mutable_param(4)->mutable_init()->set_type(kUniform);
		gru1_conf.mutable_param(5)->set_name("br1");
		gru1_conf.mutable_param(5)->set_type(kParam);
		gru1_conf.mutable_param(5)->mutable_init()->set_type(kUniform);
		gru1_conf.mutable_param(6)->set_name("wchx1");
		gru1_conf.mutable_param(6)->set_type(kParam);
		gru1_conf.mutable_param(6)->mutable_init()->set_type(kUniform);
		gru1_conf.mutable_param(7)->set_name("wchh1");
		gru1_conf.mutable_param(7)->set_type(kParam);
		gru1_conf.mutable_param(7)->mutable_init()->set_type(kUniform);
		gru1_conf.mutable_param(8)->set_name("bc1");
		gru1_conf.mutable_param(8)->set_type(kParam);
		gru1_conf.mutable_param(8)->mutable_init()->set_type(kUniform);

		gru2_conf.mutable_gru_conf() -> set_dim_hidden(2);
		gru2_conf.mutable_gru_conf() -> set_bias_term(true);
		for (int i = 0; i < 9; i ++) {
			gru2_conf.add_param();
		}
		gru2_conf.mutable_param(0)->set_name("wzhx2");
		gru2_conf.mutable_param(0)->set_share_from("wzhx1");
		gru2_conf.mutable_param(0)->set_type(kParam);
		gru2_conf.mutable_param(0)->mutable_init()->set_type(kUniform);

		gru2_conf.mutable_param(1)->set_name("wzhh2");
		gru2_conf.mutable_param(1)->set_share_from("wzhh1");
		gru2_conf.mutable_param(1)->set_type(kParam);
		gru2_conf.mutable_param(1)->mutable_init()->set_type(kUniform);

		gru2_conf.mutable_param(2)->set_name("bz2");
		gru2_conf.mutable_param(2)->set_share_from("bz1");
		gru2_conf.mutable_param(2)->set_type(kParam);
		gru2_conf.mutable_param(2)->mutable_init()->set_type(kUniform);

		gru2_conf.mutable_param(3)->set_name("wrhx2");
		gru2_conf.mutable_param(3)->set_share_from("wrhx1");
		gru2_conf.mutable_param(3)->set_type(kParam);
		gru2_conf.mutable_param(3)->mutable_init()->set_type(kUniform);

		gru2_conf.mutable_param(4)->set_name("wrhh2");
		gru2_conf.mutable_param(4)->set_share_from("wrhh1");
		gru2_conf.mutable_param(4)->set_type(kParam);
		gru2_conf.mutable_param(4)->mutable_init()->set_type(kUniform);

		gru2_conf.mutable_param(5)->set_name("br2");
		gru2_conf.mutable_param(5)->set_share_from("br1");
		gru2_conf.mutable_param(5)->set_type(kParam);
		gru2_conf.mutable_param(5)->mutable_init()->set_type(kUniform);

		gru2_conf.mutable_param(6)->set_name("wchx2");
		gru2_conf.mutable_param(6)->set_share_from("wchx1");
		gru2_conf.mutable_param(6)->set_type(kParam);
		gru2_conf.mutable_param(6)->mutable_init()->set_type(kUniform);

		gru2_conf.mutable_param(7)->set_name("wchh2");
		gru2_conf.mutable_param(7)->set_share_from("wchh1");
		gru2_conf.mutable_param(7)->set_type(kParam);
		gru2_conf.mutable_param(7)->mutable_init()->set_type(kUniform);

		gru2_conf.mutable_param(8)->set_name("bc2");
		gru2_conf.mutable_param(8)->set_share_from("bc1");
		gru2_conf.mutable_param(8)->set_type(kParam);
		gru2_conf.mutable_param(8)->mutable_init()->set_type(kUniform);

	}
	singa::LayerProto in1_conf;
	singa::LayerProto in2_conf;
	singa::LayerProto gru1_conf;
	singa::LayerProto gru2_conf;
};

TEST_F(GRULayerTest, Setup) {
	singa::Driver driver;
	//driver.RegisterLayer<GRULayer, int> (kGRU);
	driver.RegisterParam<Param>(0);
	driver.RegisterParamGenerator<UniformGen>(kUniform);
	driver.RegisterParamGenerator<ParamGenerator>(kConstant);

	singa::CSVInputLayer in_layer_1;
	singa::CSVInputLayer in_layer_2;

	in_layer_1.Setup(in1_conf, std::vector<singa::Layer*> { });
	EXPECT_EQ(2, static_cast<int>(in_layer_1.aux_data().size()));
	EXPECT_EQ(8, in_layer_1.data(nullptr).count());

	in_layer_2.Setup(in2_conf, std::vector<singa::Layer*>{ });
	EXPECT_EQ(2, static_cast<int>(in_layer_2.aux_data().size()));
	EXPECT_EQ(8, in_layer_2.data(nullptr).count());

	singa::GRULayer gru_layer_1;
	gru_layer_1.Setup(gru1_conf, std::vector<singa::Layer*>{&in_layer_1});
	EXPECT_EQ(2, gru_layer_1.hdim());
	EXPECT_EQ(4, gru_layer_1.vdim());
	gru_layer_1.GetParams()[0]->InitValues();
	cout << "gru_layer_1: " << gru_layer_1.GetParams()[0]->data().cpu_data()[0] << endl;

	//singa::GRULayer gru_layer_2;
	//gru_layer_2.Setup(gru2_conf, std::vector<singa::Layer*>{&in_layer_2, &gru_layer_1});
	//EXPECT_EQ(2, gru_layer_2.hdim());
	//EXPECT_EQ(4, gru_layer_2.vdim());
}

/*

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
 */
