#include "gtest/gtest.h"
#include "singa/model/layer.h"
#include "singa/singa_config.h"

TEST(Layer, CreateLayer) {
  std::vector<std::string> types{
      "convolution", "dense", "dropout", "relu", "batchnorm",
      "flatten",     "lrn",   "pooling", "prelu",      "softmax"};
  for (auto type : types) {
    auto layer = singa::CreateLayer("singacpp_" + type);
    // EXPECT_EQ(layer->layer_type(), type);
  }
}

#ifdef USE_CUDNN
TEST(Layer, CreateCudnnLayer) {
  std::vector<std::string> types{
      "convolution", "dropout", "relu", "batchnorm",
      "lrn",   "pooling", "softmax"};
#if CUDNN_VERSION_MAJOR >= 5
  types.push_back("dropout");
#endif
  for (auto type : types) {
    auto layer = singa::CreateLayer("cudnn_" + type);
    // EXPECT_EQ(layer->layer_type(), type);
  }
}
#endif
