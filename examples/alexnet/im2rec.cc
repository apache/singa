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


#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <cstdint>
#include <iostream>
#include <vector>

#include "singa/io/store.h"
#include "singa/proto/common.pb.h"

using std::string;

const int kImageSize = 256;
const int kImageNBytes = 256*256*3;

void create_data(const string& image_list,
    const string& input_folder,
    const string& output_folder,
    const string& backend = "kvfile") {
  singa::RecordProto image;
  image.add_shape(3);
  image.add_shape(kImageSize);
  image.add_shape(kImageSize);

  singa::RecordProto mean;
  mean.CopyFrom(image);
  for (int i = 0; i < kImageNBytes; ++i)
    mean.add_data(0.f);

  auto store = singa::io::CreateStore(backend);
  if (backend == "lmdb")
    CHECK(store->Open(output_folder + "/image_record", singa::io::kCreate));
  else
    CHECK(store->Open(output_folder + "/image_record.bin", singa::io::kCreate));

  LOG(INFO) << "Generating image record";

  std::ifstream image_list_file(image_list.c_str(), std::ios::in);
  CHECK(image_list_file.is_open()) << "Unable to open image list";

  string image_file_name;
  int label;
  char str_buffer[kImageNBytes];
  string rec_buf;
  cv::Mat img, res;
  std::vector<std::pair<string, int>> file_list;
  while (image_list_file >> image_file_name >> label)
    file_list.push_back(std::make_pair(image_file_name, label));
  LOG(INFO) << "Data Shuffling";
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(file_list.begin(), file_list.end()
      , std::default_random_engine());
  LOG(INFO) << "Total number of images is " << file_list.size();
  int ImageNum = file_list.size();

  for (int imageid = 0; imageid < ImageNum; ++imageid) {
    string path = input_folder + "/" + file_list[imageid].first;
    img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    CHECK(img.data != NULL) << "OpenCV load image fail" << path;
    cv::resize(img, res, cv::Size(kImageSize, kImageSize),
        0, 0, CV_INTER_LINEAR);
    for (int h = 0; h < kImageSize; ++h) {
      const uchar* ptr = res.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < kImageSize; ++w)
        for (int c = 0; c < 3; ++c)
          str_buffer[(c*kImageSize+h)*kImageSize+w] =
            static_cast<uint8_t>(ptr[img_index++]);
    }
    /*
    for (int i = 0; i < kImageSize; ++i) {
      for (int j = 0; j < kImageSize; ++j) {
        cv::Vec3b pixel = res.at<cv::Vec3b>(j, i);
        str_buffer[i*kImageSize+j] = static_cast<uint8_t>(pixel.val[2]);
        str_buffer[kImageSize*kImageSize+i*kImageSize+j] = static_cast<uint8_t>(pixel.val[1]);
        str_buffer[kImageSize*kImageSize*2+i*kImageSize+j] = static_cast<uint8_t>(pixel.val[0]);
      }
    }
    */
    image.set_label(file_list[imageid].second);
    image.set_pixel(str_buffer, kImageNBytes);
    image.SerializeToString(&rec_buf);

    int length = snprintf(str_buffer, kImageNBytes, "%08d", imageid);
    CHECK(store->Write(string(str_buffer, length), rec_buf));
    if ((imageid+1) % 1000 == 0) {
      store->Flush();
      LOG(INFO) << imageid+1 << " files processed.";
    }
    const string& pixels = image.pixel();
    for (int i = 0; i < kImageNBytes; ++i)
      mean.set_data(i, mean.data(i) + static_cast<uint8_t>(pixels[i]));
  }
  if (ImageNum % 1000 != 0)
      LOG(INFO) << ImageNum << " files processed.";

  store->Flush();
  store->Close();

  LOG(INFO) << "Create image mean";
  if (backend == "lmdb")
    CHECK(store->Open(output_folder + "/image_mean", singa::io::kCreate));
  else
    CHECK(store->Open(output_folder + "/image_mean.bin", singa::io::kCreate));
  for (int i = 0; i < kImageNBytes; i++)
    mean.set_data(i, mean.data(i) / ImageNum);
  mean.SerializeToString(&rec_buf);
  store->Write("mean", rec_buf);
  store->Flush();
  store->Close();
  delete store;

  LOG(INFO) << "Done!";
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cout << "Create data stores for ImageNet dataset.\n"
      << "Usage: <image_list_file> <input_image_folder> <output_folder>"
      << " <Optional: backend {lmdb, kvfile} default: kvfile>\n";
  } else {
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = 1;
    if (argc == 4)
      create_data(string(argv[1]), string(argv[2]), string(argv[3]));
    else
      create_data(string(argv[1]), string(argv[2]),
          string(argv[3]), string(argv[4]));
  }
  return 0;
}
