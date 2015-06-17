#include "utils/param.h"
#include "gtest/gtest.h"


using namespace singa;

const int param_size[]={2400,32,25600,32, 51200,64,57600,10};

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
