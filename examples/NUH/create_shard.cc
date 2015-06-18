//
// This code creates DataShard for MNIST dataset.
// It is adapted from the convert_mnist_data from Caffe
//
// Usage:
//    create_shard.bin input_image_file input_label_file output_folder
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <cstdint>
#include <iostream>

#include <fstream>
#include <string>

#include "utils/data_shard.h"
#include "utils/common.h"
#include "proto/model.pb.h"

using singa::DataShard;
using singa::WriteProtoToBinaryFile;
using std::string;


void create_shard(const char* data_filename, const char* output) {
  // Open files
  std::ifstream data_file(data_filename, std::ios::in | std::ios::binary);
  CHECK(data_file) << "Unable to open file " << data_filename;
  // Read the meta data
  uint32_t num_items;
  uint32_t num_features;


  data_file.read(reinterpret_cast<char*>(&num_items), 4);
  data_file.read(reinterpret_cast<char*>(&num_features), 4);

  DataShard shard(output, DataShard::kCreate);
  char label;
  char* features = new char[num_features];
  int count = 0;
  const int kMaxKeyLength = 10;
  char key[kMaxKeyLength];
  string value;

  singa::Record record;
  singa::SingleLabelImageRecord* data=record.mutable_image();
  data->add_shape(1);
  data->add_shape(num_features);
  LOG(INFO) << "A total of " << num_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  for (int item_id = 0; item_id < num_items; ++item_id) {
    data_file.read(features, rows * cols);
    data_file.read(&label, 1);
    data->set_pixel(features, rows*cols);
    data->set_label(label);
    snprintf(key, kMaxKeyLength, "%08d", item_id);
    shard.Insert(string(key), record);
  }
  delete features;
  shard.Flush();
}

int main(int argc, char** argv) {
/*
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage("This program create a DataShard for a MNIST dataset\n"
        "Usage:\n"
        "    create_shard.bin  input_image_file input_label_file output_db_file\n"
        "The MNIST dataset could be downloaded at\n"
        "    http://yann.lecun.com/exdb/mnist/\n"
        "You should gunzip them after downloading.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
    gflags::ShowUsageWithFlagsRestrict(argv[0],
        "examples/mnist/create_shard.bin");
*/

  if (argc != 3) {
    std::cout<<"This program create a DataShard for a MNIST dataset\n"
        "Usage:\n"
        "    create_shard.bin  input_data_file output_db_file\n";
  } else {
    google::InitGoogleLogging(argv[0]);
    create_shard(argv[1], argv[2]);
  }
  return 0;
}
