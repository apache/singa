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

#include <fstream>
#include <string>

#include "gtest/gtest.h"
#include "singa/core/tensor.h"
#include "singa/io/reader.h"
#include "singa/io/snapshot.h"

const std::string prefix = "./snapshot_test";
const float param_1_data[] = {0.1f, 0.2f, 0.3f, 0.4f};
const float param_2_data[] = {0.2f, 0.1f, 0.4f, 0.3f};
const std::string desc_1 =
    "parameter name: Param_1\tdata type: 0\tdim: 1\tshape: 4";
const std::string desc_2 =
    "parameter name: Param_2\tdata type: 0\tdim: 2\tshape: 2 2";
const int int_data[] = {1, 3, 5, 7};
const double double_data[] = {0.2, 0.4, 0.6, 0.8};

TEST(Snapshot, WriteTest) {
  singa::Snapshot snapshot(prefix, singa::Snapshot::kWrite);
  singa::Tensor param_1(singa::Shape{4}), param_2(singa::Shape{2, 2});
  param_1.CopyDataFromHostPtr(param_1_data, 4);
  param_2.CopyDataFromHostPtr(param_2_data, 4);
  snapshot.Write("Param_1", param_1);
  snapshot.Write("Param_2", param_2);
}

TEST(Snapshot, ReadTest) {
  singa::Snapshot snapshot(prefix, singa::Snapshot::kRead);
  singa::Tensor param_1, param_2;
  singa::Shape shape1, shape2;
  shape1 = snapshot.ReadShape("Param_1");
  EXPECT_EQ(shape1.size(), 1u);
  EXPECT_EQ(shape1[0], 4u);
  shape2 = snapshot.ReadShape("Param_2");
  EXPECT_EQ(shape2.size(), 2u);
  EXPECT_EQ(shape2[0], 2u);
  EXPECT_EQ(shape2[1], 2u);
  param_1 = snapshot.Read("Param_1");
  const float* data_1 = param_1.data<float>();
  for (size_t i = 0; i < singa::Product(shape1); ++i)
    EXPECT_FLOAT_EQ(data_1[i], param_1_data[i]);
  param_2 = snapshot.Read("Param_2");
  const float* data_2 = param_2.data<float>();
  for (size_t i = 0; i < singa::Product(shape2); ++i)
    EXPECT_FLOAT_EQ(data_2[i], param_2_data[i]);
  std::ifstream desc_file(prefix + ".desc");
  std::string line;
  getline(desc_file, line);
  getline(desc_file, line);
  EXPECT_EQ(line, desc_1);
  getline(desc_file, line);
  EXPECT_EQ(line, desc_2);
}

TEST(Snapshot, ReadIntTest) {
  {
    singa::Snapshot int_snapshot_write(prefix + ".int",
                                       singa::Snapshot::kWrite);
    singa::Tensor int_param(singa::Shape{4});
    int_param.CopyDataFromHostPtr(int_data, 4);
    int_param.AsType(singa::kInt);
    int_snapshot_write.Write("IntParam", int_param);
  }

  {
    singa::Snapshot int_snapshot_read(prefix + ".int", singa::Snapshot::kRead);
    singa::Shape shape;
    shape = int_snapshot_read.ReadShape("IntParam");
    EXPECT_EQ(shape.size(), 1u);
    EXPECT_EQ(shape[0], 4u);
    singa::Tensor int_param = int_snapshot_read.Read("IntParam");
    const int* param_data = int_param.data<int>();
    for (size_t i = 0; i < singa::Product(shape); ++i)
      EXPECT_EQ(param_data[i], int_data[i]);
  }
}

/*
TEST(Snapshot, ReadDoubleTest) {
  {
    singa::Snapshot double_snapshot_write(prefix + ".double",
                                          singa::Snapshot::kWrite);
    singa::Tensor double_param(singa::Shape{4});
    double_param.AsType(singa::kDouble);
    double_param.CopyDataFromHostPtr(double_data, 4);
    double_snapshot_write.Write("DoubleParam", double_param);
  }

  {
    singa::Snapshot double_snapshot_read(prefix + ".double",
                                         singa::Snapshot::kRead);
    singa::Shape shape;
    shape = double_snapshot_read.ReadShape("DoubleParam");
    EXPECT_EQ(shape.size(), 1u);
    EXPECT_EQ(shape[0], 4u);
    singa::Tensor double_param = double_snapshot_read.Read("DoubleParam");
    const double* param_data = double_param.data<double>();
    for (size_t i = 0; i < singa::Product(shape); ++i)
      EXPECT_EQ(param_data[i], double_data[i]);
  }
}
*/
