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
#include "ilsvrc12.h"
#include "singa/utils/channel.h"
#include "singa/utils/string.h"
int main(int argc, char **argv) {
  int pos = singa::ArgPos(argc, argv, "-trainlist");
  string train_image_list = "/data/xiangrui/label/train.txt";
  if (pos != -1) train_image_list = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-trainfolder");
  string train_image_folder = "/data/xiangrui/ILSVRC2012_img_train";
  if (pos != -1) train_image_folder = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-testlist");
  string test_image_list = "/data/xiangrui/label/val.txt";
  if (pos != -1) test_image_list = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-testfolder");
  string test_image_folder = "/data/xiangrui/ILSVRC2012_img_val";
  if (pos != -1) test_image_folder = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-outdata");
  string bin_folder = "/home/xiangrui/imagenet_data";
  if (pos != -1) bin_folder = argv[pos + 1];

  pos = singa::ArgPos(argc, argv, "-filesize");
  size_t train_file_size = 1280;
  if (pos != -1) train_file_size = atoi(argv[pos + 1]);
  singa::ILSVRC data;
  LOG(INFO) << "Creating training and test data...";
  data.CreateTrainData(train_image_list, train_image_folder, bin_folder,
                       train_file_size);
  data.CreateTestData(test_image_list, test_image_folder, bin_folder);
  LOG(INFO) << "Data created!";
  return 0;
}
#endif  // USE_OPENCV
