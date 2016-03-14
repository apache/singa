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
#include "singa/neuralnet/input_layer.h"
#include "singa/neuralnet/neuron_layer.h"
#include "singa/neuralnet/neuralnet.h"
#include "singa/neuralnet/connection_layer.h"
#include "singa/driver.h"
#include "singa/proto/job.pb.h"
#include "singa/utils/common.h"

using namespace singa;

class UnrollingTest: public ::testing::Test {
protected:
	virtual void SetUp() {
		NetProto* net_conf1 = job_conf1.mutable_neuralnet();

		LayerProto* data_layer1 = net_conf1->add_layer();
		data_layer1->set_name("data");
		data_layer1->set_type(kRecordInput);

		LayerProto* embedding_layer1 = net_conf1->add_layer();
		embedding_layer1->set_name("embedding");
		embedding_layer1->set_type(kDummy);
		embedding_layer1->add_srclayers("data");
		embedding_layer1->set_unroll_len(3);
		embedding_layer1->add_unroll_conn_type(kUnrollOneToAll);

		LayerProto* gru_layer1 = net_conf1->add_layer();
		gru_layer1->set_name("gru");
		gru_layer1->set_type(kGRU);
		gru_layer1->add_srclayers("embedding");
		gru_layer1->mutable_gru_conf()->set_dim_hidden(20);
		gru_layer1->add_param()->set_name("w_z_hx");
		gru_layer1->add_param()->set_name("w_r_hx");
		gru_layer1->add_param()->set_name("w_c_hx");
		gru_layer1->add_param()->set_name("w_z_hh");
		gru_layer1->add_param()->set_name("w_r_hh");
		gru_layer1->add_param()->set_name("w_c_hh");
		gru_layer1->set_unroll_len(3);
		gru_layer1->add_unroll_conn_type(kUnrollOneToOne);

		LayerProto* out_layer1 = net_conf1->add_layer();
		out_layer1->set_name("out");
		out_layer1->set_type(kInnerProduct);
		out_layer1->add_srclayers("gru");
		out_layer1->mutable_innerproduct_conf()->set_num_output(100);
		out_layer1->add_param()->set_name("w");
		out_layer1->add_param()->set_name("b");
		out_layer1->set_unroll_len(3);
		out_layer1->add_unroll_conn_type(kUnrollOneToOne);

		LayerProto* loss_layer1 = net_conf1->add_layer();
		loss_layer1->set_name("loss");
		loss_layer1->set_type(kSoftmaxLoss);
		loss_layer1->add_srclayers("out");
		loss_layer1->add_srclayers("data");
		loss_layer1->set_unroll_len(3);
		loss_layer1->add_unroll_conn_type(kUnrollOneToOne);
		loss_layer1->add_unroll_conn_type(kUnrollOneToAll);

		/*
		 * Initialize job conf 2
		NetProto* net_conf2 = job_conf2.mutable_neuralnet();

		LayerProto* data_layer2 = net_conf2->add_layer();
		data_layer2->set_name("data");
		data_layer2->set_type(kRecordInput);

		LayerProto* embedding_layer2 = net_conf2->add_layer();
		embedding_layer2->set_name("embedding");
		embedding_layer2->set_type(kDummy);
		embedding_layer2->add_srclayers("data");
		embedding_layer2->add_srclayers("softmax");
		embedding_layer2->set_unroll_len(3);
		embedding_layer2->add_unroll_conn_type(kUnrollOneToAll);
		embedding_layer2->add_shift(0);
		embedding_layer2->add_unroll_conn_type(kUnrollOneToOne);
		embedding_layer2->add_shift(1);

		LayerProto* gru_layer2 = net_conf2->add_layer();
		gru_layer2->set_name("gru");
		gru_layer2->set_type(kGRU);
		gru_layer2->add_srclayers("embedding");
		gru_layer2->mutable_gru_conf()->set_dim_hidden(20);
		gru_layer2->mutable_gru_conf()->set_bias_term(false);
		gru_layer2->add_param()->set_name("w_z_hx");
		gru_layer2->add_param()->set_name("w_r_hx");
		gru_layer2->add_param()->set_name("w_c_hx");
		gru_layer2->add_param()->set_name("w_z_hh");
		gru_layer2->add_param()->set_name("w_r_hh");
		gru_layer2->add_param()->set_name("w_c_hh");
		gru_layer2->set_unroll_len(3);
		gru_layer2->add_unroll_conn_type(kUnrollOneToOne);
		gru_layer2->add_shift(0);

		LayerProto* out_layer2 = net_conf2->add_layer();
		out_layer2->set_name("out");
		out_layer2->set_type(kInnerProduct);
		out_layer2->add_srclayers("gru");
		out_layer2->mutable_innerproduct_conf()->set_num_output(100);
		out_layer2->add_param()->set_name("w");
		out_layer2->add_param()->set_name("b");
		out_layer2->set_unroll_len(3);
		out_layer2->add_unroll_conn_type(kUnrollOneToOne);
		out_layer2->add_shift(0);

		LayerProto* softmax_layer2 = net_conf2->add_layer();
		softmax_layer2->set_name("softmax");
		softmax_layer2->set_type(kSoftmax);
		softmax_layer2->add_srclayers("out");
		softmax_layer2->set_unroll_len(3);
		softmax_layer2->add_unroll_conn_type(kUnrollOneToOne);
		softmax_layer2->add_shift(0);

		LayerProto* loss_layer2 = net_conf2->add_layer();
		loss_layer2->set_name("loss");
		loss_layer2->set_type(kSoftmaxLoss);
		loss_layer2->add_srclayers("softmax");
		loss_layer2->add_srclayers("data");
		loss_layer2->set_unroll_len(3);
		loss_layer2->add_unroll_conn_type(kUnrollOneToOne);
		loss_layer2->add_shift(0);
		loss_layer2->add_unroll_conn_type(kUnrollOneToAll);
		loss_layer2->add_shift(0);
		 */
	}

	singa::JobProto job_conf1;
	singa::JobProto job_conf2;
};

TEST_F(UnrollingTest, GRULanguageModelTrain) {
	NetProto net;
	net.CopyFrom(job_conf1.neuralnet());
	NetProto unrolled_net = NeuralNet::Unrolling(net);
	EXPECT_EQ("0#data", unrolled_net.layer(0).name());

	EXPECT_EQ("0#embedding", unrolled_net.layer(1).name());
	EXPECT_EQ(1, unrolled_net.layer(1).srclayers_size());
	EXPECT_EQ("0#data", unrolled_net.layer(1).srclayers(0));

	EXPECT_EQ("1#embedding", unrolled_net.layer(2).name());
	EXPECT_EQ(1, unrolled_net.layer(2).srclayers_size());
	EXPECT_EQ("0#data", unrolled_net.layer(2).srclayers(0));

	EXPECT_EQ("2#embedding", unrolled_net.layer(3).name());
	EXPECT_EQ(1, unrolled_net.layer(3).srclayers_size());
	EXPECT_EQ("0#data", unrolled_net.layer(3).srclayers(0));

	EXPECT_EQ("0#gru", unrolled_net.layer(4).name());
	EXPECT_EQ(1, unrolled_net.layer(4).srclayers_size());
	EXPECT_EQ("0#embedding", unrolled_net.layer(4).srclayers(0));
	EXPECT_EQ("0#w_z_hx", unrolled_net.layer(4).param(0).name());
	EXPECT_EQ("0#w_r_hx", unrolled_net.layer(4).param(1).name());
	EXPECT_EQ("0#w_c_hx", unrolled_net.layer(4).param(2).name());
	EXPECT_EQ("0#w_z_hh", unrolled_net.layer(4).param(3).name());
	EXPECT_EQ("0#w_r_hh", unrolled_net.layer(4).param(4).name());
	EXPECT_EQ("0#w_c_hh", unrolled_net.layer(4).param(5).name());

	EXPECT_EQ("1#gru", unrolled_net.layer(5).name());
	EXPECT_EQ(2, unrolled_net.layer(5).srclayers_size());
	EXPECT_EQ("1#embedding", unrolled_net.layer(5).srclayers(0));
	EXPECT_EQ("0#gru", unrolled_net.layer(5).srclayers(1));
	EXPECT_EQ("1#w_z_hx", unrolled_net.layer(5).param(0).name());
	EXPECT_EQ("0#w_z_hx", unrolled_net.layer(5).param(0).share_from());
	EXPECT_EQ("1#w_r_hx", unrolled_net.layer(5).param(1).name());
	EXPECT_EQ("0#w_r_hx", unrolled_net.layer(5).param(1).share_from());
	EXPECT_EQ("1#w_c_hx", unrolled_net.layer(5).param(2).name());
	EXPECT_EQ("0#w_c_hx", unrolled_net.layer(5).param(2).share_from());
	EXPECT_EQ("1#w_z_hh", unrolled_net.layer(5).param(3).name());
	EXPECT_EQ("0#w_z_hh", unrolled_net.layer(5).param(3).share_from());
	EXPECT_EQ("1#w_r_hh", unrolled_net.layer(5).param(4).name());
	EXPECT_EQ("0#w_r_hh", unrolled_net.layer(5).param(4).share_from());
	EXPECT_EQ("1#w_c_hh", unrolled_net.layer(5).param(5).name());
	EXPECT_EQ("0#w_c_hh", unrolled_net.layer(5).param(5).share_from());

	EXPECT_EQ("2#gru", unrolled_net.layer(6).name());
	EXPECT_EQ(2, unrolled_net.layer(6).srclayers_size());
	EXPECT_EQ("2#embedding", unrolled_net.layer(6).srclayers(0));
	EXPECT_EQ("1#gru", unrolled_net.layer(6).srclayers(1));
	EXPECT_EQ("2#w_z_hx", unrolled_net.layer(6).param(0).name());
	EXPECT_EQ("0#w_z_hx", unrolled_net.layer(6).param(0).share_from());
	EXPECT_EQ("2#w_r_hx", unrolled_net.layer(6).param(1).name());
	EXPECT_EQ("0#w_r_hx", unrolled_net.layer(6).param(1).share_from());
	EXPECT_EQ("2#w_c_hx", unrolled_net.layer(6).param(2).name());
	EXPECT_EQ("0#w_c_hx", unrolled_net.layer(6).param(2).share_from());
	EXPECT_EQ("2#w_z_hh", unrolled_net.layer(6).param(3).name());
	EXPECT_EQ("0#w_z_hh", unrolled_net.layer(6).param(3).share_from());
	EXPECT_EQ("2#w_r_hh", unrolled_net.layer(6).param(4).name());
	EXPECT_EQ("0#w_r_hh", unrolled_net.layer(6).param(4).share_from());
	EXPECT_EQ("2#w_c_hh", unrolled_net.layer(6).param(5).name());
	EXPECT_EQ("0#w_c_hh", unrolled_net.layer(6).param(5).share_from());

	EXPECT_EQ("0#out", unrolled_net.layer(7).name());
	EXPECT_EQ(1, unrolled_net.layer(7).srclayers_size());
	EXPECT_EQ("0#gru", unrolled_net.layer(7).srclayers(0));
	EXPECT_EQ("0#w", unrolled_net.layer(7).param(0).name());
	EXPECT_EQ("0#b", unrolled_net.layer(7).param(1).name());

	EXPECT_EQ("1#out", unrolled_net.layer(8).name());
	EXPECT_EQ(1, unrolled_net.layer(8).srclayers_size());
	EXPECT_EQ("1#gru", unrolled_net.layer(8).srclayers(0));
	EXPECT_EQ("1#w", unrolled_net.layer(8).param(0).name());
	EXPECT_EQ("0#w", unrolled_net.layer(8).param(0).share_from());
	EXPECT_EQ("1#b", unrolled_net.layer(8).param(1).name());
	EXPECT_EQ("0#b", unrolled_net.layer(8).param(1).share_from());

	EXPECT_EQ("2#out", unrolled_net.layer(9).name());
	EXPECT_EQ(1, unrolled_net.layer(9).srclayers_size());
	EXPECT_EQ("2#gru", unrolled_net.layer(9).srclayers(0));
	EXPECT_EQ("2#w", unrolled_net.layer(9).param(0).name());
	EXPECT_EQ("0#w", unrolled_net.layer(9).param(0).share_from());
	EXPECT_EQ("2#b", unrolled_net.layer(9).param(1).name());
	EXPECT_EQ("0#b", unrolled_net.layer(9).param(1).share_from());

	EXPECT_EQ("0#loss", unrolled_net.layer(10).name());
	EXPECT_EQ(2, unrolled_net.layer(10).srclayers_size());
	EXPECT_EQ("0#out", unrolled_net.layer(10).srclayers(0));
	EXPECT_EQ("0#data", unrolled_net.layer(10).srclayers(1));

	EXPECT_EQ("1#loss", unrolled_net.layer(11).name());
	EXPECT_EQ(2, unrolled_net.layer(11).srclayers_size());
	EXPECT_EQ("1#out", unrolled_net.layer(11).srclayers(0));
	EXPECT_EQ("0#data", unrolled_net.layer(11).srclayers(1));

	EXPECT_EQ("2#loss", unrolled_net.layer(12).name());
	EXPECT_EQ(2, unrolled_net.layer(12).srclayers_size());
	EXPECT_EQ("2#out", unrolled_net.layer(12).srclayers(0));
	EXPECT_EQ("0#data", unrolled_net.layer(12).srclayers(1));
}

/*
TEST_F(UnrollingTest, GRULanguageModelTest) {
	NetProto net;
	net.CopyFrom(job_conf2.neuralnet());
	NetProto unrolled_net = NeuralNet::Unrolling(net);

	EXPECT_EQ("data", unrolled_net.layer(0).name());

	EXPECT_EQ("0#embedding", unrolled_net.layer(1).name());
	EXPECT_EQ(1, unrolled_net.layer(1).srclayers_size());
	EXPECT_EQ("data", unrolled_net.layer(1).srclayers(0));

	EXPECT_EQ("1#embedding", unrolled_net.layer(2).name());
	EXPECT_EQ(2, unrolled_net.layer(2).srclayers_size());
	EXPECT_EQ("data", unrolled_net.layer(2).srclayers(0));
	EXPECT_EQ("0#softmax", unrolled_net.layer(2).srclayers(1));

	EXPECT_EQ("2#embedding", unrolled_net.layer(3).name());
	EXPECT_EQ(2, unrolled_net.layer(3).srclayers_size());
	EXPECT_EQ("data", unrolled_net.layer(3).srclayers(0));
	EXPECT_EQ("1#softmax", unrolled_net.layer(3).srclayers(1));

	EXPECT_EQ("0#gru", unrolled_net.layer(4).name());
	EXPECT_EQ(1, unrolled_net.layer(4).srclayers_size());
	EXPECT_EQ("0#embedding", unrolled_net.layer(4).srclayers(0));
	EXPECT_EQ("w_z_hx", unrolled_net.layer(4).param(0).name());
	EXPECT_EQ("w_r_hx", unrolled_net.layer(4).param(1).name());
	EXPECT_EQ("w_c_hx", unrolled_net.layer(4).param(2).name());
	EXPECT_EQ("w_z_hh", unrolled_net.layer(4).param(3).name());
	EXPECT_EQ("w_r_hh", unrolled_net.layer(4).param(4).name());
	EXPECT_EQ("w_c_hh", unrolled_net.layer(4).param(5).name());

	EXPECT_EQ("1#gru", unrolled_net.layer(5).name());
	EXPECT_EQ(2, unrolled_net.layer(5).srclayers_size());
	EXPECT_EQ("0#gru", unrolled_net.layer(5).srclayers(0));
	EXPECT_EQ("1#embedding", unrolled_net.layer(5).srclayers(1));
	EXPECT_EQ("1#w_z_hx", unrolled_net.layer(5).param(0).name());
	EXPECT_EQ("w_z_hx", unrolled_net.layer(5).param(0).share_from());
	EXPECT_EQ("1#w_r_hx", unrolled_net.layer(5).param(1).name());
	EXPECT_EQ("w_r_hx", unrolled_net.layer(5).param(1).share_from());
	EXPECT_EQ("1#w_c_hx", unrolled_net.layer(5).param(2).name());
	EXPECT_EQ("w_c_hx", unrolled_net.layer(5).param(2).share_from());
	EXPECT_EQ("1#w_z_hh", unrolled_net.layer(5).param(3).name());
	EXPECT_EQ("w_z_hh", unrolled_net.layer(5).param(3).share_from());
	EXPECT_EQ("1#w_r_hh", unrolled_net.layer(5).param(4).name());
	EXPECT_EQ("w_r_hh", unrolled_net.layer(5).param(4).share_from());
	EXPECT_EQ("1#w_c_hh", unrolled_net.layer(5).param(5).name());
	EXPECT_EQ("w_c_hh", unrolled_net.layer(5).param(5).share_from());

	EXPECT_EQ("2#gru_2", unrolled_net.layer(6).name());
	EXPECT_EQ(2, unrolled_net.layer(6).srclayers_size());
	EXPECT_EQ("1#gru", unrolled_net.layer(6).srclayers(0));
	EXPECT_EQ("2#embedding", unrolled_net.layer(6).srclayers(1));
	EXPECT_EQ("2#w_z_hx", unrolled_net.layer(6).param(0).name());
	EXPECT_EQ("w_z_hx", unrolled_net.layer(6).param(0).share_from());
	EXPECT_EQ("2#w_r_hx", unrolled_net.layer(6).param(1).name());
	EXPECT_EQ("w_r_hx", unrolled_net.layer(6).param(1).share_from());
	EXPECT_EQ("2#w_c_hx", unrolled_net.layer(6).param(2).name());
	EXPECT_EQ("w_c_hx", unrolled_net.layer(6).param(2).share_from());
	EXPECT_EQ("2#w_z_hh", unrolled_net.layer(6).param(3).name());
	EXPECT_EQ("w_z_hh", unrolled_net.layer(6).param(3).share_from());
	EXPECT_EQ("2#w_r_hh", unrolled_net.layer(6).param(4).name());
	EXPECT_EQ("w_r_hh", unrolled_net.layer(6).param(4).share_from());
	EXPECT_EQ("2#w_c_hh", unrolled_net.layer(6).param(5).name());
	EXPECT_EQ("w_c_hh", unrolled_net.layer(6).param(5).share_from());

	EXPECT_EQ("out_0", unrolled_net.layer(7).name());
	EXPECT_EQ(1, unrolled_net.layer(7).srclayers_size());
	EXPECT_EQ("gru_0", unrolled_net.layer(7).srclayers(0));
	EXPECT_EQ("w", unrolled_net.layer(7).param(0).name());
	EXPECT_EQ("b", unrolled_net.layer(7).param(1).name());

	EXPECT_EQ("out_1", unrolled_net.layer(8).name());
	EXPECT_EQ(1, unrolled_net.layer(8).srclayers_size());
	EXPECT_EQ("gru_1", unrolled_net.layer(8).srclayers(0));
	EXPECT_EQ("w_1", unrolled_net.layer(8).param(0).name());
	EXPECT_EQ("w", unrolled_net.layer(8).param(0).share_from());
	EXPECT_EQ("b_1", unrolled_net.layer(8).param(1).name());
	EXPECT_EQ("b", unrolled_net.layer(8).param(1).share_from());

	EXPECT_EQ("out_2", unrolled_net.layer(9).name());
	EXPECT_EQ(1, unrolled_net.layer(9).srclayers_size());
	EXPECT_EQ("gru_2", unrolled_net.layer(9).srclayers(0));
	EXPECT_EQ("w_2", unrolled_net.layer(9).param(0).name());
	EXPECT_EQ("w", unrolled_net.layer(9).param(0).share_from());
	EXPECT_EQ("b_2", unrolled_net.layer(9).param(1).name());
	EXPECT_EQ("b", unrolled_net.layer(9).param(1).share_from());

	EXPECT_EQ("softmax_0", unrolled_net.layer(10).name());
	EXPECT_EQ(1, unrolled_net.layer(10).srclayers_size());
	EXPECT_EQ("out_0", unrolled_net.layer(10).srclayers(0));

	EXPECT_EQ("softmax_1", unrolled_net.layer(11).name());
	EXPECT_EQ(1, unrolled_net.layer(11).srclayers_size());
	EXPECT_EQ("out_1", unrolled_net.layer(11).srclayers(0));

	EXPECT_EQ("softmax_2", unrolled_net.layer(12).name());
	EXPECT_EQ(1, unrolled_net.layer(12).srclayers_size());
	EXPECT_EQ("out_2", unrolled_net.layer(12).srclayers(0));

	EXPECT_EQ("loss_0", unrolled_net.layer(13).name());
	EXPECT_EQ(2, unrolled_net.layer(13).srclayers_size());
	EXPECT_EQ("softmax_0", unrolled_net.layer(13).srclayers(0));
	EXPECT_EQ("data", unrolled_net.layer(13).srclayers(1));

	EXPECT_EQ("loss_1", unrolled_net.layer(14).name());
	EXPECT_EQ(2, unrolled_net.layer(14).srclayers_size());
	EXPECT_EQ("softmax_1", unrolled_net.layer(14).srclayers(0));
	EXPECT_EQ("data", unrolled_net.layer(14).srclayers(1));

	EXPECT_EQ("loss_2", unrolled_net.layer(15).name());
	EXPECT_EQ(2, unrolled_net.layer(15).srclayers_size());
	EXPECT_EQ("softmax_2", unrolled_net.layer(15).srclayers(0));
	EXPECT_EQ("data", unrolled_net.layer(15).srclayers(1));
}
  */
