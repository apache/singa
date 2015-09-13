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
 * The some functions in this file are adapted from Caffe whose license
 * is attached.
 *
 * COPYRIGHT
 * All contributions by the University of California:
 * Copyright (c) 2014, The Regents of the University of California (Regents)
 * All rights reserved.
 * All other contributions:
 * Copyright (c) 2014, the respective contributors
 * All rights reserved.
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 * LICENSE
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * CONTRIBUTION AGREEMENT
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 */

#include "utils/common.h"

#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>

#include <stdarg.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <cfloat>

#include <glog/logging.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace singa {

using std::string;
using std::vector;
const int kBufLen = 1024;

string IntVecToString(const vector<int>& vec) {
  string disp = "(";
  for (int x : vec)
    disp += std::to_string(x) + ", ";
  return disp + ")";
}

/**
 *  * Formatted string.
 *   */
string VStringPrintf(string fmt, va_list l) {
  char buffer[4096];
  vsnprintf(buffer, sizeof(buffer), fmt.c_str(), l);
  return string(buffer);
}

/**
 *  * Formatted string.
 *   */
string StringPrintf(string fmt, ...) {
  va_list l;
  va_start(l, fmt);  // fmt.AsString().c_str());
  string result = VStringPrintf(fmt, l);
  va_end(l);
  return result;
}

int ArgPos(int argc, char** arglist, const char* arg) {
  for (int i = 0; i < argc; i++) {
    if (strcmp(arglist[i], arg) == 0) {
      return i;
    }
  }
  return -1;
}

void  CreateFolder(const string name) {
  struct stat buffer;
  if (stat(name.c_str(), &buffer) != 0) {
    mkdir(name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    CHECK_EQ(stat(name.c_str(), &buffer), 0);
  }
}

const vector<vector<int>> Slice(int num, const vector<int>& sizes) {
  vector<vector<int>> slices;
  if (num == 0)
    return slices;
  int avg = 0;
  for (int x : sizes)
      avg += x;
  avg = avg / num + avg % num;
  int diff = avg / 10;
  LOG(INFO) << "Slicer, param avg = " << avg << ", diff = " << diff;

  int capacity = avg, nbox = 0;
  for (int x : sizes) {
    vector<int> slice;
    string slicestr = "";
    while (x > 0) {
      int size = 0;
      if (capacity >= x) {
        capacity -= x;
        size = x;
        x = 0;
      } else if (capacity + diff >= x) {
        size = x;
        x = 0;
        capacity = 0;
      } else if (capacity >= diff) {
        x -= capacity;
        size = capacity;
        capacity = avg;
        nbox++;
      } else {
        capacity = avg;
        nbox++;
      }
      if (size) {
        slice.push_back(size);
        slicestr += ", " + std::to_string(size);
      }
    }
    LOG(INFO) << slicestr;
    slices.push_back(slice);
  }
  CHECK_LE(nbox, num);
  return slices;
}

const vector<int> PartitionSlices(int num, const vector<int>& slices) {
  vector<int> slice2box;
  if (num == 0)
    return slice2box;
  int avg = 0;
  for (int x : slices)
    avg += x;
  avg = avg / num + avg % num;
  int box = avg, boxid = 0, diff = avg / 10;
  for (auto it = slices.begin(); it != slices.end();) {
    int x = *it;
    if (box >= x) {
      box -= x;
      slice2box.push_back(boxid);
      it++;
    } else if (box + diff >= x) {
      slice2box.push_back(boxid);
      it++;
      box = 0;
    } else {
      box = avg;
      boxid++;
    }
  }
  CHECK_EQ(slice2box.size(), slices.size());
  int previd = -1;
  string disp;
  for (size_t i = 0; i < slice2box.size(); i++) {
    if (previd != slice2box[i]) {
      previd = slice2box[i];
      disp += " box = " +std::to_string(previd) + ":";
    }
    disp += " " + std::to_string(slices[i]);
  }
  LOG(INFO) << "partition slice (avg = " << avg
            << ", num = " << num << "):" << disp;
  return slice2box;
}

int gcd(int a, int b) {
  for (;;) {
    if (a == 0) return b;
    b %= a;
    if (b == 0) return a;
    a %= b;
  }
}

int LeastCommonMultiple(int a, int b) {
  int temp = gcd(a, b);
  return temp ? (a / temp * b) : 0;
}

string GetHostIP() {
  int fd;
  struct ifreq ifr;
  fd = socket(AF_INET, SOCK_DGRAM, 0);
  /* I want to get an IPv4 IP address */
  ifr.ifr_addr.sa_family = AF_INET;
  /* I want IP address attached to "eth0" */
  strncpy(ifr.ifr_name, "eth0", IFNAMSIZ-1);
  ioctl(fd, SIOCGIFADDR, &ifr);
  close(fd);
  string ip(inet_ntoa(((struct sockaddr_in *)&ifr.ifr_addr)->sin_addr));
  /* display result */
  LOG(INFO) << "Host IP=(" << ip;
  return ip;
}

void SetupLog(const string& log_dir, const string& model) {
  // TODO(wangwei) check if NFS, then create folder using script, otherwise
  // may have problems due to multiple processes create the same folder.
  CreateFolder(log_dir);
  string warn = log_dir + "/" + model + "-warn-";
  string info = log_dir + "/" +  model + "-info-";
  string error = log_dir + "/" +  model + "-error-";
  string fatal = log_dir + "/" + model + "-fatal-";
  google::SetLogDestination(google::WARNING, warn.c_str());
  google::SetLogDestination(google::INFO, info.c_str());
  google::SetLogDestination(google::ERROR, error.c_str());
  google::SetLogDestination(google::FATAL, fatal.c_str());
}

Metric::Metric(const string& str) {
  ParseFrom(str);
}

void Metric::Add(const string& name, float value) {
  Add( name, value, 1);
}
void Metric::Add(const string& name, float value, int count) {
  if (entry_.find(name) == entry_.end()) {
    entry_[name] = std::make_pair(1, value);
  } else {
    auto& e = entry_.at(name);
    e.first += count;
    e.second += value;
  }
}

void Metric::Reset() {
  for (auto& e : entry_) {
    e.second.first = 0;
    e.second.second = 0;
  }
}

string Metric::ToLogString() const {
  string ret;
  size_t k = 0;
  for (auto e : entry_) {
    ret += e.first + " : ";
    ret += std::to_string(e.second.second / e.second.first);
    if (++k < entry_.size())
      ret += ", ";
  }
  return ret;
}

string Metric::ToString() const {
  MetricProto proto;
  for (auto e : entry_) {
    proto.add_name(e.first);
    proto.add_count(e.second.first);
    proto.add_val(e.second.second);
  }
  string ret;
  proto.SerializeToString(&ret);
  return ret;
}

void Metric::ParseFrom(const string& msg) {
  MetricProto proto;
  proto.ParseFromString(msg);
  Reset();
  for (int i = 0; i < proto.name_size(); i++) {
    entry_[proto.name(i)] = std::make_pair(proto.count(i), proto.val(i));
  }
}


/*************Below functions are adapted from Caffe ************/
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;


void Im2col(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* data_col) {
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

void Col2im(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* data_im) {
  memset(data_im, 0, height * width * channels * sizeof(float));
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int channels_col = channels * patch_h * patch_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % patch_w;
    int h_offset = (c / patch_w) % patch_h;
    int c_im = c / patch_h / patch_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
            data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

void ForwardMaxPooling(const float* bottom, const int num, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* top, float* mask) {
  int top_height = (height + pad_h * 2 -kernel_h) / stride_h + 1;
  int top_width = (width + pad_w * 2 -kernel_w) / stride_w + 1;
  int top_count = num * top_height * top_width * channels;
  for (int i = 0; i < top_count; i++) {
    mask[i] = -1;
    top[i] = -FLT_MAX;
  }
  const int bottom_offset =  height * width;
  const int top_offset = top_height * top_width;
  // The main loop
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < top_height; ++ph) {
        for (int pw = 0; pw < top_width; ++pw) {
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = std::min(hstart + kernel_h, height);
          int wend = std::min(wstart + kernel_w, width);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          const int top_index = ph * top_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width + w;
              if (bottom[index] > top[top_index]) {
                top[top_index] = bottom[index];
                mask[top_index] = index;
              }
            }
          }
        }
      }
      // compute offset
      bottom += bottom_offset;
      top += top_offset;
      mask += top_offset;
    }
  }
}

void BackwardMaxPooling(const float* top, const float* mask, const int num,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* bottom) {
  int top_height = (height + pad_h * 2 -kernel_h) / stride_h + 1;
  int top_width = (width + pad_w * 2 -kernel_w) / stride_w + 1;
  const int top_offset = top_height * top_width;
  const int bottom_offset = height * width;
  memset(bottom, 0, sizeof(float) * num * channels * bottom_offset);
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < top_height; ++ph) {
        for (int pw = 0; pw < top_width; ++pw) {
          const int top_idx = ph * top_width + pw;
          const int bottom_idx = static_cast<int>(mask[top_idx]);
          bottom[bottom_idx] += top[top_idx];
        }
      }
      top += top_offset;
      mask += top_offset;
      bottom += bottom_offset;
    }
  }
}

void ForwardAvgPooling(const float* bottom, const int num, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* top) {
  int top_height = (height + pad_h * 2 -kernel_h) / stride_h + 1;
  int top_width = (width + pad_w * 2 -kernel_w) / stride_w + 1;
  int top_count = num * top_height * top_width * channels;
  for (int i = 0; i < top_count; i++) {
    top[i] = 0;
  }
  const int bottom_offset =  height * width;
  const int top_offset = top_height * top_width;
  // The main loop
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < top_height; ++ph) {
        for (int pw = 0; pw < top_width; ++pw) {
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = std::min(hstart + kernel_h, height+pad_h);
          int wend = std::min(wstart + kernel_w, width+pad_w);
          int pool_size = (hend-hstart) * (wend-wstart);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          hend = std::min(hend, height);
          wend = std::min(wend, width);
          const int top_index = ph * top_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width + w;
              top[top_index] += bottom[index];
            }
          }
          top[top_index] /= pool_size;
        }
      }
      // compute offset
      bottom += bottom_offset;
      top += top_offset;
    }
  }
}

void BackwardAvgPooling(const float* top, const int num, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* bottom) {
  int top_height = (height + pad_h * 2 -kernel_h) / stride_h + 1;
  int top_width = (width + pad_w * 2 -kernel_w) / stride_w + 1;
  const int top_offset = top_height * top_width;
  const int bottom_offset = height * width;
  memset(bottom, 0, sizeof(float) * num * channels * bottom_offset);
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < top_height; ++ph) {
        for (int pw = 0; pw < top_width; ++pw) {
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = std::min(hstart + kernel_h, height+pad_h);
          int wend = std::min(wstart + kernel_w, width+pad_w);
          int pool_size = (hend-hstart) * (wend-wstart);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          hend = std::min(hend, height);
          wend = std::min(wend, width);
          const int top_index = ph * top_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width + w;
              bottom[index] += top[top_index] / pool_size;
            }
          }
        }
      }
      top += top_offset;
      bottom += bottom_offset;
    }
  }
}

void ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  CHECK(google::protobuf::TextFormat::Parse(input, proto));
  delete input;
  close(fd);
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

void ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  // upper limit 512MB, warning threshold 256MB
  coded_input->SetTotalBytesLimit(536870912, 268435456);
  CHECK(proto->ParseFromCodedStream(coded_input));
  delete coded_input;
  delete raw_input;
  close(fd);
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_CREAT|O_WRONLY|O_TRUNC, 0644);
  CHECK_NE(fd, -1) << "File cannot open: " << filename;
  CHECK(proto.SerializeToFileDescriptor(fd));
}
}  // namespace singa
