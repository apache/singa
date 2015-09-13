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

#ifndef SINGA_UTILS_COMMON_H_
#define SINGA_UTILS_COMMON_H_

#include <google/protobuf/message.h>
#include <unordered_map>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include "proto/common.pb.h"

namespace singa {

std::string IntVecToString(const std::vector<int>& vec);
std::string VStringPrintf(std::string fmt, va_list l);
std::string StringPrintf(std::string fmt, ...);

/**
 * Locate the position of the arg in arglist.
 *
 * @param argc total num of arguments
 * @param arglist all arguments
 * @param the searched argument
 * @return the position of arg in the arglist; -1 if not found.
 */
int ArgPos(int argc, char** arglist, const char* arg);
void CreateFolder(const std::string name);
/**
 * Slice a set of large Params into small pieces such that they can be roughtly
 * equally partitioned into a fixed number of boxes.
 *
 * @param num total number of boxes to store the small pieces
 * @param sizes size of all Params
 * @return all slices for each Param
 */
const std::vector<std::vector<int>> Slice(int num,
    const std::vector<int>& sizes);
/**
 * Partition slices into boxes.
 *
 * @param num number of boxes
 * @param slices slice sizes
 * @return box id for each slice
 */
const std::vector<int> PartitionSlices(int num, const std::vector<int>& slices);
/*
inline void Sleep(int millisec=1){
  std::this_thread::sleep_for(std::chrono::milliseconds(millisec));
}
*/
int gcd(int a, int b);
int LeastCommonMultiple(int a, int b);
/*
inline float rand_real() {
  return  static_cast<float>(rand_r())/(RAND_MAX+1.0f);
}
*/
std::string GetHostIP();
void SetupLog(const std::string& workspace, const std::string& model);

/**
 * Performance mtrics.
 */
class Metric {
 public:
  Metric() {}
  explicit Metric(const std::string& str);
  /**
   * Add one metric.
   *
   * If the metric exist, the aggregate. Otherwise create a new entry for it.
   *
   * @param name metric name, e.g., 'loss'
   * @param value metric value
   */
  void Add(const std::string& name, float value);
  void Add(const std::string& name, float value, int count);
  /**
   * reset all metric counter and value to 0
   */
  void Reset();
  /**
   * Generate a one-line string for logging
   */
  std::string ToLogString() const;
  /**
   * Serialize the object into a string
   */
  std::string ToString() const;
  /**
   * Parse the metric from a string
   */
  void ParseFrom(const std::string& msg);

 private:
  std::unordered_map<std::string, std::pair<int, float>> entry_;
};

using google::protobuf::Message;
void Im2col(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* data_col);
void Col2im(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* data_im);
void ForwardMaxPooling(const float* bottom, const int num, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* top, float* mask);
void BackwardMaxPooling(const float* top, const float* mask, const int num,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* bottom);
void ForwardAvgPooling(const float* bottom, const int num, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* top);
void BackwardAvgPooling(const float* top, const int num, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* bottom);

void ReadProtoFromTextFile(const char* filename, Message* proto);
void WriteProtoToTextFile(const Message& proto, const char* filename);
void ReadProtoFromBinaryFile(const char* filename, Message* proto);
void WriteProtoToBinaryFile(const Message& proto, const char* filename);


}  // namespace singa

#endif  // SINGA_UTILS_COMMON_H_
