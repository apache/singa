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

#include "singa/comm/msg.h"

#include <glog/logging.h>
#include <stdarg.h>

namespace singa {

Msg::~Msg() {
#ifdef USE_ZMQ
  if (msg_ != nullptr)
    zmsg_destroy(&msg_);
  frame_ = nullptr;
#else
  for (auto& frame : frames_)
    delete static_cast<char*>(frame.first);
#endif
}

Msg::Msg() {
#ifdef USE_ZMQ
  msg_ = zmsg_new();
#endif
}

Msg::Msg(const Msg& msg) {
  src_ = msg.src_;
  dst_ = msg.dst_;
  type_ = msg.type_;
  trgt_val_ = msg.trgt_val_;
  trgt_version_ = msg.trgt_version_;
#ifdef USE_ZMQ
  msg_ = zmsg_dup(msg.msg_);
#endif
}

Msg::Msg(int src, int dst) {
  src_ = src;
  dst_ = dst;
#ifdef USE_ZMQ
  msg_ = zmsg_new();
#endif
}

void Msg::SwapAddr() {
  std::swap(src_, dst_);
}

#ifdef USE_ZMQ
int Msg::size() const {
  return zmsg_content_size(msg_);
}
void Msg::AddFrame(const void* addr, int nBytes) {
  zmsg_addmem(msg_, addr, nBytes);
}
int Msg::FrameSize() {
  return zframe_size(frame_);
}
char* Msg::FrameStr() {
  return zframe_strdup(frame_);
}
void* Msg::FrameData() {
  return zframe_data(frame_);
}
bool Msg::NextFrame() {
  frame_ = zmsg_next(msg_);
  return frame_ != nullptr;
}
void Msg::FirstFrame() {
  frame_ = zmsg_first(msg_);
}
void Msg::LastFrame() {
  frame_ = zmsg_last(msg_);
}
void Msg::ParseFromZmsg(zmsg_t* msg) {
  char* tmp = zmsg_popstr(msg);
  sscanf(tmp, "%d %d %d %d %d",
         &src_, &dst_, &type_, &trgt_val_, &trgt_version_);
  frame_ = zmsg_first(msg);
  msg_ = msg;
}

zmsg_t* Msg::DumpToZmsg() {
  zmsg_pushstrf(msg_, "%d %d %d %d %d",
      src_, dst_, type_, trgt_val_, trgt_version_);
  zmsg_t *tmp = msg_;
  msg_ = nullptr;
  return tmp;
}

#else

int Msg::size() const {
  int s = 0;
  for (auto& entry : frames_)
    s += entry.second;
  return s;
}

void Msg::AddFrame(const void* addr, int nBytes) {
  char* tmp = new char[nBytes];
  memcpy(tmp, addr, nBytes);
  frames_.push_back(std::make_pair(tmp, nBytes));
}

int Msg::FrameSize() {
  return frames_.at(idx_).second;
}

char* Msg::FrameStr() {
  char* ret = new char[frames_.at(idx_).second];
  memcpy(ret, static_cast<char*>(frames_.at(idx_).first), 
        frames_.at(idx_).second);
  return ret;
}

void* Msg::FrameData() {
  return frames_.at(idx_).first;
}

bool Msg::NextFrame() {
  idx_++;
//  LOG(ERROR) << "idx " << idx_ << " vs size " << frames_.size();
  return idx_ < frames_.size();
}

void Msg::FirstFrame() {
  idx_ = 0;
}

void Msg::LastFrame() {
  idx_ = frames_.size() - 1;
}

#endif

// frame marker indicating this frame is serialize like printf
#define FMARKER "*singa*"

#define kMaxFrameLen 2048

int Msg::AddFormatFrame(const char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  int size = strlen(FMARKER);
  char dst[kMaxFrameLen];
  memcpy(dst, FMARKER, size);
  dst[size++] = 0;
  while (*format) {
    if (*format == 'i') {
      int x = va_arg(argptr, int);
      dst[size++] = 'i';
      memcpy(dst + size, &x, sizeof(x));
      size += sizeof(x);
    } else if (*format == 'f') {
      float x = static_cast<float> (va_arg(argptr, double));
      dst[size++] = 'f';
      memcpy(dst + size, &x, sizeof(x));
      size += sizeof(x);
    } else if (*format == '1') {
      uint8_t x = va_arg(argptr, int);
      memcpy(dst + size, &x, sizeof(x));
      size += sizeof(x);
    } else if (*format == '2') {
      uint16_t x = va_arg(argptr, int);
      memcpy(dst + size, &x, sizeof(x));
      size += sizeof(x);
    } else if (*format == '4') {
      uint32_t x = va_arg(argptr, uint32_t);
      memcpy(dst + size, &x, sizeof(x));
      size += sizeof(x);
    } else if (*format == 's') {
      char* x = va_arg(argptr, char *);
      dst[size++] = 's';
      memcpy(dst + size, x, strlen(x));
      size += strlen(x);
      dst[size++] = 0;
    } else if (*format == 'p') {
      void* x = va_arg(argptr, void *);
      dst[size++] = 'p';
      memcpy(dst + size, &x, sizeof(x));
      size += sizeof(x);
    } else {
      LOG(ERROR) << "Unknown format " << *format;
    }
    format++;
    CHECK_LE(size, kMaxFrameLen);
  }
  va_end(argptr);
  AddFrame(dst, size);
  return size;
}

int Msg::ParseFormatFrame(const char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  char* src = FrameStr();
  CHECK_STREQ(FMARKER, src);
  int size = strlen(FMARKER) + 1;
  while (*format) {
    if (*format == 'i') {
      int *x = va_arg(argptr, int *);
      CHECK_EQ(src[size++], 'i');
      memcpy(x, src + size, sizeof(*x));
      size += sizeof(*x);
    } else if (*format == 'f') {
      float *x = va_arg(argptr, float *);
      CHECK_EQ(src[size++], 'f');
      memcpy(x, src + size, sizeof(*x));
      size += sizeof(*x);
    } else if (*format == '1') {
      uint8_t *x = va_arg(argptr, uint8_t *);
      memcpy(x, src + size, sizeof(*x));
      size += sizeof(*x);
    } else if (*format == '2') {
      uint16_t *x = va_arg(argptr, uint16_t *);
      memcpy(x, src + size, sizeof(*x));
      size += sizeof(*x);
    } else if (*format == '4') {
      uint32_t *x = va_arg(argptr, uint32_t *);
      memcpy(x, src + size, sizeof(*x));
      size += sizeof(*x);
    } else if (*format == 's') {
      char* x = va_arg(argptr, char *);
      CHECK_EQ(src[size++], 's');
      int len = strlen(src + size);
      memcpy(x, src + size, len);
      x[len] = 0;
      size += len + 1;
    } else if (*format == 'p') {
      void** x = va_arg(argptr, void **);
      CHECK_EQ(src[size++], 'p');
      memcpy(x, src + size, sizeof(*x));
      size += sizeof(*x);
    } else {
      LOG(ERROR) << "Unknown format type " << *format;
    }
    format++;
  }
  va_end(argptr);
  //  delete src;
  return size;
}

}  // namespace singa
