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


/**
 * Create training and test DataShard for CIFAR dataset. 
 * It is adapted from convert_cifar_data from Caffe. 
 *    create_shard.bin <input> <output_folder> 
 * 
 * Read from JobConf object the option to use KVfile, HDFS or other (1st layer
 * store_conf object). 
 * To load to HDFS, specify "hdfs://namenode/examples" as the output folder
 */

#include <glog/logging.h>
#include <fstream>
#include <string>
#include <cstdint>
#include <iostream>

#include "singa/io/store.h"
#include "singa/proto/common.pb.h"
#include "singa/utils/common.h"

using std::string;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARBatchSize = 10000;
const int kCIFARTrainBatches = 5;

void read_image(std::ifstream* file, int* label, char* buffer) {
  char label_char;
  file->read(&label_char, 1);
  *label = label_char;
  file->read(buffer, kCIFARImageNBytes);
  return;
}

void create_data(const string& input_folder, const string& output_folder) {
  int label;
  char str_buffer[kCIFARImageNBytes];
  string rec_buf;
  singa::RecordProto image;
  image.add_shape(3);
  image.add_shape(kCIFARSize);
  image.add_shape(kCIFARSize);

  singa::RecordProto mean;
  mean.CopyFrom(image);
  for (int i = 0; i < kCIFARImageNBytes; i++)
    mean.add_data(0.f);

  string store_backend = output_folder.find("hdfs") !=-1 ?
                         "hdfsfile" : "kvfile";
  auto store = singa::io::CreateStore(store_backend);
  CHECK(store->Open(output_folder + "/train_data.bin", singa::io::kCreate));
  LOG(INFO) << "Preparing training data";
  int count = 0;
  for (int fileid = 0; fileid < kCIFARTrainBatches; ++fileid) {
    LOG(INFO) << "Training Batch " << fileid + 1;
    snprintf(str_buffer, kCIFARImageNBytes, "/data_batch_%d.bin", fileid + 1);
    std::ifstream data_file((input_folder + str_buffer).c_str(),
        std::ios::in | std::ios::binary);
    CHECK(data_file.is_open()) << "Unable to open train file #" << fileid + 1;
    for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
      read_image(&data_file, &label, str_buffer);
      image.set_label(label);
      image.set_pixel(str_buffer, kCIFARImageNBytes);
      image.SerializeToString(&rec_buf);
      int length = snprintf(str_buffer, kCIFARImageNBytes, "%05d", count);
      CHECK(store->Write(string(str_buffer, length), rec_buf));

      const string& pixels = image.pixel();
      for (int i = 0; i < kCIFARImageNBytes; i++)
        mean.set_data(i, mean.data(i) + static_cast<uint8_t>(pixels[i]));
      count += 1;
    }
  }
  store->Flush();
  store->Close();

  LOG(INFO) << "Create image mean";
  store->Open(output_folder + "/image_mean.bin", singa::io::kCreate);
  for (int i = 0; i < kCIFARImageNBytes; i++)
    mean.set_data(i, mean.data(i) / count);
  mean.SerializeToString(&rec_buf);
  store->Write("mean", rec_buf);
  store->Flush();
  store->Close();

  LOG(INFO) << "Create test data";
  store->Open(output_folder + "/test_data.bin", singa::io::kCreate);
  std::ifstream data_file((input_folder + "/test_batch.bin").c_str(),
      std::ios::in | std::ios::binary);
  CHECK(data_file.is_open()) << "Unable to open test file.";
  for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
    read_image(&data_file, &label, str_buffer);
    image.set_label(label);
    image.set_pixel(str_buffer, kCIFARImageNBytes);
    image.SerializeToString(&rec_buf);
    int length = snprintf(str_buffer, kCIFARImageNBytes, "%05d", itemid);
    CHECK(store->Write(string(str_buffer, length), rec_buf));
  }
  store->Flush();
  store->Close();
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout <<"Create train and test DataShard for Cifar dataset.\n"
      << "Usage:\n"
      << "    create_data.bin input_folder output_folder\n"
      << "Where the input folder should contain the binary batch files.\n";
  } else {
    google::InitGoogleLogging(argv[0]);
    create_data(string(argv[1]), string(argv[2]));
  }
  return 0;
}
