#include "gtest/gtest.h"
#include "singa/model/layer.h"
#include "singa/singa_config.h"

TEST(Layer, CreateLayer) {
  std::vector<std::string> types{
      "Convolution", "Dense", "Dropout", "Activation", "BatchNorm",
      "Flatten",     "LRN",   "Pooling", "PReLU",      "Softmax"};
  for (auto type : types) {
    auto layer = singa::CreateLayer(type);
    EXPECT_EQ(layer->layer_type(), type);
  }
}

#ifdef USE_CUDNN
TEST(Layer, CreateCudnnLayer) {
  std::vector<std::string> types{
      "CudnnConvolution", "CudnnActivation",
      "CudnnBatchNorm",   "Flatten",      "CudnnLRN",
      "CudnnPooling",     "PReLU",        "CudnnSoftmax"};
#if CUDNN_VERSION_MAJOR >= 5
  types.push_back("CudnnDropout");
#endif
  for (auto type : types) {
    auto layer = singa::CreateLayer(type);
    EXPECT_EQ(layer->layer_type(), type);
  }
}
#endif
