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

#include "singa/utils/param.h"
#include "gtest/gtest.h"


using namespace singa;

const int param_size[] = {2400, 32, 25600, 32, 51200, 64, 57600, 10};

/*
class ParamSlicerTest : public ::testing::Test {
  public:
    ParamSlicerTest() {
      ParamProto proto;
      int nparams=sizeof(param_size)/sizeof(int);
      for(int i=0;i<nparams;i++){
        vector<int> shape{param_size[i]};
        auto param=std::make_shared<Param>();
        param->Setup(proto, shape);
        param->set_id(i);
        params.push_back(param);
      }
    }
  protected:
    vector<shared_ptr<Param>> params;
};

// all params are stored in one box, no need to split
TEST_F(ParamSlicerTest, OneBox){
  int nparams=sizeof(param_size)/sizeof(int);
  ParamSlicer slicer;
  int num=1;
  auto slices=slicer.Slice(num, params);
  ASSERT_EQ(slices.size(),nparams);
  ASSERT_EQ(slicer.Get(1).size(),1);
  ASSERT_EQ(slicer.Get(2).size(),1);
  ASSERT_EQ(slicer.Get(nparams-1).back(), slices.size()-1);
}

// there are multiple boxes
TEST_F(ParamSlicerTest, MultipleBox){
  int nparams=sizeof(param_size)/sizeof(int);
  ParamSlicer slicer;
  int num=4;
  auto slices=slicer.Slice(num, params);
  ASSERT_EQ(slicer.Get(1).size(),1);
  ASSERT_EQ(slicer.Get(3).size(),1);
  ASSERT_EQ(slicer.Get(nparams-1).back(), slices.size()-1);
}
*/
