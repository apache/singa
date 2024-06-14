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
#include "singa/singa_config.h"
#ifdef USE_OPENCV
#ifndef SINGA_EXAMPLES_IMAGENET_ILSVRC12_H_
#define SINGA_EXAMPLES_IMAGENET_ILSVRC12_H_
#include <omp.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <thread>
#include <vector>

#include "singa/core/tensor.h"
#include "singa/io/decoder.h"
#include "singa/io/encoder.h"
#include "singa/io/reader.h"
#include "singa/io/transformer.h"
#include "singa/io/writer.h"
#include "singa/proto/io.pb.h"
#include "singa/utils/timer.h"

using std::string;
using namespace singa::io;
namespace singa {

/// size for resizing
const size_t kImageSize = 256;
const size_t kImageNBytes = 3 * kImageSize * kImageSize;
/// size for cropping
const size_t kCropSize = 227;
/// For reading ILSVRC2012 image data as tensors.
class ILSVRC {
 public:
  /// Setup encoder, decoder
  ILSVRC();
  ~ILSVRC() {
    if (encoder != nullptr) delete encoder;
    if (decoder != nullptr) delete decoder;
    if (transformer != nullptr) delete transformer;
    if (reader != nullptr) {
      reader->Close();
      delete reader;
    }
    if (writer != nullptr) {
      writer->Close();
      delete writer;
    }
  }
  /// Create binary files for training data
  /// train_image_list: list file of training images
  /// train_image_folder: folder where stores original training images
  /// train_bin_folder: folder to store binary files
  /// train_file_size: number of images that are contain in one binary file
  void CreateTrainData(string train_image_list, string train_image_folder,
                       string train_bin_folder, size_t train_file_size);
  /// Create binary files for test data
  /// train_image_list: list file of test images
  /// train_image_folder: folder where saves original test images
  /// train_bin_folder: folder to save binary files
  void CreateTestData(string test_image_list, string test_image_folder,
                      string test_bin_folder);
  /// Load data from a binary file,  return <images, labels> pair
  /// suppose the data will be loaded file by file.
  /// flag: kTrain or kTest
  /// file: binary file which stores the images
  /// read_size: number of images to be loaded
  /// offset: offset in the file
  /// n_read: number of images which are read
  size_t LoadData(int flag, string file, size_t read_size, Tensor *x, Tensor *y,
                  size_t *n_read, int nthreads);
  /// A wrapper method to spawn a thread to execute LoadData() method.
  std::thread AsyncLoadData(int flag, string file, size_t read_size, Tensor *x,
                            Tensor *y, size_t *n_read, int nthreads);

  void DecodeTransform(int flag, int thid, int nthreads,
                       vector<string *> images, Tensor *x, Tensor *y);
  /// A wrapper method to spawn a thread to execute Decodetransform() method.
  std::thread AsyncDecodeTransform(int flag, int thid, int nthreads,
                                   vector<string *> images, Tensor *x,
                                   Tensor *y);

  /// Read mean from path
  void ReadMean(string path);

 protected:
  /// Read one image at path, resize the image
  Tensor ReadImage(string path);
  /// Write buff to the file in kCreate/kAppend mode
  void Write(string outfile, singa::io::Mode mode);
  void WriteMean(Tensor &mean, string path);

 private:
  Tensor mean;
  string last_read_file = "";

  JPGEncoder *encoder = nullptr;
  JPGDecoder *decoder = nullptr;
  ImageTransformer *transformer = nullptr;
  BinFileReader *reader = nullptr;
  BinFileWriter *writer = nullptr;
};

ILSVRC::ILSVRC() {
  EncoderConf en_conf;
  en_conf.set_image_dim_order("CHW");
  encoder = new JPGEncoder();
  encoder->Setup(en_conf);

  DecoderConf de_conf;
  de_conf.set_image_dim_order("CHW");
  decoder = new JPGDecoder();
  decoder->Setup(de_conf);

  TransformerConf trans_conf;
  trans_conf.add_crop_shape(kCropSize);
  trans_conf.add_crop_shape(kCropSize);
  trans_conf.set_image_dim_order("CHW");
  trans_conf.set_horizontal_mirror(true);
  transformer = new ImageTransformer();
  transformer->Setup(trans_conf);
}

Tensor ILSVRC::ReadImage(string path) {
  cv::Mat mat = cv::imread(path, CV_LOAD_IMAGE_COLOR);
  CHECK(mat.data != NULL) << "OpenCV load image fail: " << path;
  cv::Size size(kImageSize, kImageSize);
  cv::Mat resized;
  cv::resize(mat, resized, size);
  CHECK_EQ((size_t)resized.size().height, kImageSize);
  CHECK_EQ((size_t)resized.size().width, kImageSize);
  // dimension_order: CHW
  Shape shape{(size_t)resized.channels(), (size_t)resized.rows,
              (size_t)resized.cols};
  Tensor image(shape, singa::kUChar);
  unsigned char *data = new unsigned char[kImageNBytes];
  for (int i = 0; i < resized.rows; i++)
    for (int j = 0; j < resized.cols; j++)
      for (int k = 0; k < resized.channels(); k++)
        data[k * kImageSize * kImageSize + i * kImageSize + j] =
            resized.at<cv::Vec3b>(i, j)[k];
  image.CopyDataFromHostPtr<unsigned char>(data, kImageNBytes);
  delete[] data;

  return image;
}

void ILSVRC::WriteMean(Tensor &mean, string path) {
  Tensor mean_lb(Shape{1}, kInt);
  std::vector<Tensor> input;
  input.push_back(mean);
  input.push_back(mean_lb);
  BinFileWriter bfwriter;
  bfwriter.Open(path, kCreate);
  bfwriter.Write(path, encoder->Encode(input));
  bfwriter.Flush();
  bfwriter.Close();
}

void ILSVRC::CreateTrainData(string image_list, string input_folder,
                             string output_folder, size_t file_size = 12800) {
  std::vector<std::pair<string, int>> file_list;
  size_t *sum = new size_t[kImageNBytes];
  for (size_t i = 0; i < kImageNBytes; i++) sum[i] = 0u;
  string image_file_name;
  int label;
  string outfile;
  std::ifstream image_list_file(image_list.c_str(), std::ios::in);
  while (image_list_file >> image_file_name >> label)
    file_list.push_back(std::make_pair(image_file_name, label));
  LOG(INFO) << "Data Shuffling";
  std::shuffle(file_list.begin(), file_list.end(),
               std::default_random_engine());
  LOG(INFO) << "Total number of training images is " << file_list.size();
  size_t num_train_images = file_list.size();
  if (file_size == 0) file_size = num_train_images;
  for (size_t imageid = 0; imageid < num_train_images; imageid++) {
    string path = input_folder + "/" + file_list[imageid].first;
    Tensor image = ReadImage(path);
    auto image_data = image.data<unsigned char>();
    for (size_t i = 0; i < kImageNBytes; i++)
      sum[i] += static_cast<size_t>(image_data[i]);
    label = file_list[imageid].second;
    Tensor lb(Shape{1}, kInt);
    lb.CopyDataFromHostPtr<int>(&label, 1);
    std::vector<Tensor> input;
    input.push_back(image);
    input.push_back(lb);
    string encoded_str = encoder->Encode(input);
    if (writer == nullptr) {
      writer = new BinFileWriter();
      outfile = output_folder + "/train" +
                std::to_string(imageid / file_size + 1) + ".bin";
      writer->Open(outfile, kCreate);
    }
    writer->Write(path, encoded_str);
    if ((imageid + 1) % file_size == 0) {
      writer->Flush();
      writer->Close();
      LOG(INFO) << "Write " << file_size << " images into " << outfile;
      delete writer;
      writer = nullptr;
    }
  }
  if (writer != nullptr) {
    writer->Flush();
    writer->Close();
    LOG(INFO) << "Write " << num_train_images % file_size << " images into "
              << outfile;
    delete writer;
    writer = nullptr;
  }
  size_t num_file =
      num_train_images / file_size + ((num_train_images % file_size) ? 1 : 0);
  LOG(INFO) << "Write " << num_train_images << " images into " << num_file
            << " binary files";
  Tensor mean = Tensor(Shape{3, kImageSize, kImageSize}, kUChar);
  unsigned char *mean_data = new unsigned char[kImageNBytes];
  for (size_t i = 0; i < kImageNBytes; i++)
    mean_data[i] = static_cast<unsigned char>(sum[i] / num_train_images);
  mean.CopyDataFromHostPtr<unsigned char>(mean_data, kImageNBytes);
  string mean_path = output_folder + "/mean.bin";
  WriteMean(mean, mean_path);
  delete[] mean_data;
  delete[] sum;
}

void ILSVRC::CreateTestData(string image_list, string input_folder,
                            string output_folder) {
  std::vector<std::pair<string, int>> file_list;
  string image_file_name;
  string outfile = output_folder + "/test.bin";
  int label;
  std::ifstream image_list_file(image_list.c_str(), std::ios::in);
  while (image_list_file >> image_file_name >> label)
    file_list.push_back(std::make_pair(image_file_name, label));
  LOG(INFO) << "Total number of test images is " << file_list.size();
  size_t num_test_images = file_list.size();
  for (size_t imageid = 0; imageid < num_test_images; imageid++) {
    string path = input_folder + "/" + file_list[imageid].first;
    Tensor image = ReadImage(path);
    label = file_list[imageid].second;
    Tensor lb(Shape{1}, singa::kInt);
    lb.CopyDataFromHostPtr<int>(&label, 1);
    std::vector<Tensor> input;
    input.push_back(image);
    input.push_back(lb);
    string encoded_str = encoder->Encode(input);
    if (writer == nullptr) {
      writer = new BinFileWriter();
      writer->Open(outfile, kCreate);
    }
    writer->Write(path, encoded_str);
  }
  if (writer != nullptr) {
    writer->Flush();
    writer->Close();
    delete writer;
    writer = nullptr;
  }
  LOG(INFO) << "Write " << num_test_images << " images into " << outfile;
}

void ILSVRC::ReadMean(string path) {
  BinFileReader bfreader;
  string key, value;
  bfreader.Open(path);
  bfreader.Read(&key, &value);
  auto ret = decoder->Decode(value);
  bfreader.Close();
  mean = ret[0];
}

std::thread ILSVRC::AsyncLoadData(int flag, string file, size_t read_size,
                                  Tensor *x, Tensor *y, size_t *n_read,
                                  int nthreads) {
  return std::thread(
      [=]() { LoadData(flag, file, read_size, x, y, n_read, nthreads); });
}

size_t ILSVRC::LoadData(int flag, string file, size_t read_size, Tensor *x,
                        Tensor *y, size_t *n_read, int nthreads) {
  if (file != last_read_file) {
    if (reader != nullptr) {
      reader->Close();
      delete reader;
      reader = nullptr;
    }
    reader = new BinFileReader();
    reader->Open(file, 100 << 20);
    last_read_file = file;
  } else if (reader == nullptr) {
    reader = new BinFileReader();
    reader->Open(file, 100 << 20);
  }
  vector<string *> images;
  for (size_t i = 0; i < read_size; i++) {
    string image_path;
    string *image = new string();
    bool ret = reader->Read(&image_path, image);
    if (ret == false) {
      reader->Close();
      delete reader;
      reader = nullptr;
      break;
    }
    images.push_back(image);
  }
  int nimg = images.size();
  *n_read = nimg;

  vector<std::thread> threads;
  for (int i = 1; i < nthreads; i++) {
    threads.push_back(AsyncDecodeTransform(flag, i, nthreads, images, x, y));
  }
  DecodeTransform(flag, 0, nthreads, images, x, y);
  for (size_t i = 0; i < threads.size(); i++) threads[i].join();
  for (int k = 0; k < nimg; k++) delete images.at(k);
  return nimg;
}

std::thread ILSVRC::AsyncDecodeTransform(int flag, int thid, int nthreads,
                                         vector<string *> images, Tensor *x,
                                         Tensor *y) {
  return std::thread(
      [=]() { DecodeTransform(flag, thid, nthreads, images, x, y); });
}

void ILSVRC::DecodeTransform(int flag, int thid, int nthreads,
                             vector<string *> images, Tensor *x, Tensor *y) {
  int nimg = images.size();
  int start = nimg / nthreads * thid;
  int end = start + nimg / nthreads;
  for (int k = start; k < end; k++) {
    std::vector<Tensor> pair = decoder->Decode(*images.at(k));
    auto tmp_image = pair[0] - mean;
    Tensor aug_image = transformer->Apply(flag, tmp_image);
    CopyDataToFrom(x, aug_image, aug_image.Size(), k * aug_image.Size());
    CopyDataToFrom(y, pair[1], 1, k);
  }
  if (thid == 0) {
    for (int k = nimg / nthreads * nthreads; k < nimg; k++) {
      std::vector<Tensor> pair = decoder->Decode(*images.at(k));
      auto tmp_image = pair[0] - mean;
      Tensor aug_image = transformer->Apply(flag, tmp_image);
      CopyDataToFrom(x, aug_image, aug_image.Size(), k * aug_image.Size());
      CopyDataToFrom(y, pair[1], 1, k);
    }
  }
}
}  // namespace singa

#endif  // SINGA_EXAMPLES_IMAGENET_ILSVRC12_H_
#endif  // USE_OPENCV
