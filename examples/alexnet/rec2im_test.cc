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

void generate_image(const string& output_folder,
    const string& key,
    const string& val) {
  float image_buf[kImageNBytes];
  singa::RecordProto image;
  image.ParseFromString(val);
  cv::Mat img = cv::Mat::zeros(kImageSize, kImageSize, CV_8UC3);
  string pixel = image.pixel();
  int label = image.label();
  string image_name = output_folder+"/"+key+"_"+std::to_string(label)+".jpg";
  std::cout << "Writing to " << image_name << "...\n";
  for (int h = 0; h < kImageSize; ++h) {
    uchar* ptr = img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < kImageSize; ++w) {
      for (int c = 0; c < 3; ++c)
        ptr[img_index++] =
          static_cast<uchar>(
              static_cast<uint8_t>(
                pixel[(c * kImageSize + h) * kImageSize + w]));
    }
  }

  cv::imwrite(image_name, img);
}

void visualize(const string& input_file,
    const string& output_folder,
    const string& id_list) {
  auto store = singa::io::OpenStore("kvfile", input_file,
      singa::io::kRead);

  std::vector<int> image_id_list;

  std::ifstream id_list_file(id_list.c_str(), std::ios::in);
  CHECK(id_list_file.is_open()) << "Unable to open image id list";
  string id_;
  while (id_list_file >> id_) {
    int x;
    x = std::stoi(id_);
    image_id_list.push_back(x);
  }
  std::sort(image_id_list.begin(), image_id_list.end());

  string key, val;
  for (int i = 0; i < image_id_list[0]; ++i)
    if (!store->Read(&key, &val)) {
      store->SeekToFirst();
      CHECK(store->Read(&key, &val));
    }
  generate_image(output_folder, key, val);

  for (size_t i = 1; i != image_id_list.size(); ++i) {
    for (int j = 0; j < image_id_list[i]-image_id_list[i-1]; ++j)
      if (!store->Read(&key, &val)) {
        store->SeekToFirst();
        CHECK(store->Read(&key, &val));
      }
    generate_image(output_folder, key, val);
  }
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << "Visualize images from binary kvfile records.\n"
      << "Usage: <input_file> <output_folder> <id_list>\n";
  } else {
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = 1;
    visualize(string(argv[1]), string(argv[2]), string(argv[3]));
  }

  return 0;
}
