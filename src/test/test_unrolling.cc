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
#include "singa/neuralnet/input_layer/record.h"
#include "singa/neuralnet/neuron_layer/inner_product.h"
#include "singa/neuralnet/neuron_layer/sigmoid.h"
#include "singa/neuralnet/loss_layer/euclidean.h"
#include "singa/neuralnet/neuralnet.h"
#include "singa/neuralnet/input_layer/csv.h"
#include "singa/driver.h"
#include "singa/proto/job.pb.h"
#include "singa/utils/common.h"

using namespace singa;

class UnrollingTest: public ::testing::Test {
protected:
	virtual void SetUp() {
		ReadProtoFromTextFile("src/test/unroll-test-1.conf", &job_conf);
	}

	singa::JobProto job_conf;
};

TEST_F(UnrollingTest, Train) {
	singa::Driver driver;
	driver.RegisterLayer<GRULayer, int> (kGRU);
	driver.RegisterLayer<RecordInputLayer,int>(kRecordInput);
	driver.RegisterLayer<InnerProductLayer,int>(kInnerProduct);
	driver.RegisterLayer<SigmoidLayer,int>(kSigmoid);
	driver.RegisterLayer<EuclideanLossLayer,int>(kEuclideanLoss);

	JobProto job;
	job.CopyFrom(job_conf);
	NeuralNet* net = NeuralNet::Create(job_conf.neuralnet(), kTrain, 1);
	//cout << net->layers().size() << endl;
}

