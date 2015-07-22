//
// This code creates training and test DataShard for CIFAR dataset.
// It is adapted from the convert_cifar_data from Caffe
//
// Usage:
//    create_shard.bin input_folder output_folder
//
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html
//

#include <fstream>
#include <string>

#include <glog/logging.h>
#include <cstdint>
#include <iostream>

#include "singa.h"

using std::string;

using singa::DataShard;
using singa::WriteProtoToBinaryFile;

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

void create_shard(const string& input_folder, const string& output_folder) {
  int label;
  // Data buffer
  char str_buffer[kCIFARImageNBytes];
  singa::Record record;
  singa::SingleLabelImageRecord* image=record.mutable_image();;
  image->add_shape(3);
  image->add_shape(kCIFARSize);
  image->add_shape(kCIFARSize);

  singa::SingleLabelImageRecord mean;
  mean.CopyFrom(*image);
  for(int i=0;i<kCIFARImageNBytes;i++)
    mean.add_data(0.);

  DataShard train_shard(output_folder+"/cifar10_train_shard",DataShard::kCreate);
  LOG(INFO) << "Writing Training data";
  int count=0;
  for (int fileid = 0; fileid < kCIFARTrainBatches; ++fileid) {
    // Open files
    LOG(INFO) << "Training Batch " << fileid + 1;
    snprintf(str_buffer, kCIFARImageNBytes, "/data_batch_%d.bin", fileid + 1);
    std::ifstream data_file((input_folder + str_buffer).c_str(),
        std::ios::in | std::ios::binary);
    CHECK(data_file) << "Unable to open train file #" << fileid + 1;
    for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
      read_image(&data_file, &label, str_buffer);
      image->set_label(label);
      image->set_pixel(str_buffer, kCIFARImageNBytes);
      int length = snprintf(str_buffer, kCIFARImageNBytes, "%05d",
          fileid * kCIFARBatchSize + itemid);
      CHECK(train_shard.Insert(string(str_buffer, length), record));

      const string& pixels=image->pixel();
      for(int i=0;i<kCIFARImageNBytes;i++)
        mean.set_data(i, mean.data(i)+static_cast<uint8_t>(pixels[i]));
      count+=1;
    }
  }
  train_shard.Flush();
  for(int i=0;i<kCIFARImageNBytes;i++)
    mean.set_data(i, mean.data(i)/count);
  WriteProtoToBinaryFile(mean, (output_folder+"/image_mean.bin").c_str());

  LOG(INFO) << "Writing Testing data";
  DataShard test_shard(output_folder+"/cifar10_test_shard",DataShard::kCreate);
  // Open files
  std::ifstream data_file((input_folder + "/test_batch.bin").c_str(),
      std::ios::in | std::ios::binary);
  CHECK(data_file) << "Unable to open test file.";
  for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
    read_image(&data_file, &label, str_buffer);
    image->set_label(label);
    image->set_pixel(str_buffer, kCIFARImageNBytes);
    int length = snprintf(str_buffer, kCIFARImageNBytes, "%05d", itemid);
    CHECK(test_shard.Insert(string(str_buffer, length), record));
  }
  test_shard.Flush();
}

int main(int argc, char** argv) {
  if (argc != 3) {
  std::cout<<"Create train and test DataShard for Cifar dataset.\n"
           <<"Usage:\n"
           <<"    create_shard.bin input_folder output_folder\n"
           <<"Where the input folder should contain the binary batch files.\n";
  } else {
    google::InitGoogleLogging(argv[0]);
    create_shard(string(argv[1]), string(argv[2]));
  }
  return 0;
}
