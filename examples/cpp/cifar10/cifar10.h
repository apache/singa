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
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "singa/core/tensor.h"
using std::string;
namespace singa {
/// For reading cifar10 binary data as tensors.
class Cifar10 {
 public:
  /// 'dir_path': path to the folder including the *.bin files
  Cifar10(string dir_path, bool normalize = true) : dir_path_(dir_path) {}

  /// read all training data into an image Tensor and a label Tensor
  const std::pair<Tensor, Tensor> ReadTrainData();
  /// read all test data into an image Tensor and a label Tensor
  const std::pair<Tensor, Tensor> ReadTestData();
  /// read data from one file into an image Tensor and a label Tensor
  const std::pair<Tensor, Tensor> ReadFile(string file);

  void ReadImage(std::ifstream* file, int* label, char* buffer);

 private:
  static const size_t kImageSize = 32;
  static const size_t kImageVol = 3072;
  static const size_t kBatchSize = 10000;
  const size_t kTrainFiles = 5;

  string dir_path_;
};

void Cifar10::ReadImage(std::ifstream* file, int* label, char* buffer) {
  char label_char;
  file->read(&label_char, 1);
  *label = static_cast<int>(label_char);
  file->read(buffer, kImageVol);
  return;
}
const std::pair<Tensor, Tensor> Cifar10::ReadFile(string file) {
  Tensor images(Shape{kBatchSize, 3, kImageSize, kImageSize});
  Tensor labels(Shape{kBatchSize}, kInt);
  if (dir_path_.back() != '/') dir_path_.push_back('/');
  LOG(INFO) << "Reading file " << dir_path_ + file;
  std::ifstream data_file((dir_path_ + file).c_str(),
                          std::ios::in | std::ios::binary);
  CHECK(data_file.is_open()) << "Unable to open file " << dir_path_ + file;
  int label;
  char image[kImageVol];
  float float_image[kImageVol];
  int tmplabels[kBatchSize];
  for (size_t itemid = 0; itemid < kBatchSize; ++itemid) {
    // LOG(INFO) << "reading " << itemid << "-th image";
    ReadImage(&data_file, &label, image);
    for (size_t i = 0; i < kImageVol; i++)
      float_image[i] = static_cast<float>(static_cast<uint8_t>(image[i]));
    images.CopyDataFromHostPtr(float_image, kImageVol, itemid * kImageVol);
    tmplabels[itemid] = label;
  }
  labels.CopyDataFromHostPtr(tmplabels, kBatchSize);
  return std::make_pair(images, labels);
}

const std::pair<Tensor, Tensor> Cifar10::ReadTrainData() {
  Tensor images(Shape{kBatchSize * kTrainFiles, 3, kImageSize, kImageSize});
  Tensor labels(Shape{kBatchSize * kTrainFiles}, kInt);
  for (size_t fileid = 0; fileid < kTrainFiles; ++fileid) {
    string file = "data_batch_" + std::to_string(fileid + 1) + ".bin";
    const auto ret = ReadFile(file);
    CopyDataToFrom(&images, ret.first, ret.first.Size(),
                   fileid * ret.first.Size());
    CopyDataToFrom(&labels, ret.second, kBatchSize, fileid * kBatchSize);
  }
  return std::make_pair(images, labels);
}
const std::pair<Tensor, Tensor> Cifar10::ReadTestData() {
  return ReadFile("test_batch.bin");
}
}  // namespace singa
