#include "gtest/gtest.h"
#include "singa/model/layer.h"

TEST(Layer, CreateLayer) {
  std::vector<std::string> types{
      "Convolution", "Dense", "Dropout", "Activation", "Batchnorm",
      "Flatten",     "LRN",   "Pooling", "PReLU",      "Softmax"};
  for (auto type : types) {
    singa::Layer* layer = singa::CreateLayer(type);
    EXPECT_EQ(layer->layer_type(), type);
  }
}

#ifdef USE_CUDNN
TEST(Layer, CreateCudnnLayer) {
  std::vector<std::string> types{
      "CudnnConvolution", "CudnnDropout", "CudnnActivation",
      "CudnnBatchnorm",   "Flatten",      "CudnnLRN",
      "CudnnPooling",     "PReLU",        "CudnnSoftmax"};
  for (auto type : types) {
    singa::Layer* layer = singa::CreateLayer(type);
    EXPECT_EQ(layer->layer_type(), type);
  }
}
#endif
