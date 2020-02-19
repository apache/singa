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

#include "gtest/gtest.h"
#include "singa/model/metric.h"

TEST(Accuracy, Compute) {
  singa::Accuracy acc;
  singa::Tensor p(singa::Shape{2, 3});
  singa::Tensor t(singa::Shape{2}, singa::kInt);
  const float pdat[6] = {0.1f, 0.3f, 0.6f, 0.3f, 0.2f, 0.5f};
  const int tdat[2] = {1, 2};  // one wrong, one correct
  p.CopyDataFromHostPtr(pdat, sizeof(pdat) / sizeof(float));
  t.CopyDataFromHostPtr(tdat, sizeof(tdat) / sizeof(int));
  float a = acc.Evaluate(p, t);
  EXPECT_FLOAT_EQ(a, 0.5f);
}
