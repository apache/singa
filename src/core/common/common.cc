/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef DISABLE_WARNINGS

#include "singa/core/common.h"
#include "singa/core/device.h"
#include <iostream>
#include <fstream>
#include <string>

namespace singa {

void* Block::mutable_data() {
    initialized_ = true;

    //Append block info: opt_type, ptr, time_stamp
    if (ptr_device_!=nullptr){
      stringstream strm2;
      strm2<<this;
      string temp_str2 = strm2.str();
      stringstream strm4;
      auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
      strm4<<t2;
      string temp_str4 = strm4.str();
      string temp = "Mutable "+temp_str2+" "+temp_str4;   
      ptr_device_->AppendInfo(temp);
    }

    //update ptr after swap in done, if variable is not swapped back yet as expected.
    if (data_ == nullptr) {
      auto tempData_ = ptr_device_->UpdateGpuPtrInfo(this);
      return static_cast<char*>(tempData_) + offset_;
    }
    
    return static_cast<char*>(data_) + offset_;
  }


const void* Block::data() const {
    CHECK(initialized_) << "Must initialize data before reading it";

    //Append block info: opt_type, ptr, time_stamp
    if (ptr_device_!=nullptr){
      //Append info.
      stringstream strm2;
      strm2<<this;
      string temp_str2 = strm2.str();
      stringstream strm4;
      auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
      strm4<<t2;
      string temp_str4 = strm4.str();
      string temp = "Read "+temp_str2+" "+temp_str4;
      ptr_device_->AppendInfo(temp);
    }

    //update ptr after swap in done, if variable is not swapped back yet as expected.
    if (data_ == nullptr) {
      auto tempData_ = ptr_device_->UpdateGpuPtrInfo(this);
      return static_cast<char*>(tempData_) + offset_;
    }

    return static_cast<char*>(data_) + offset_;
  }

void* Block::get_data() {
  //get data without calling data(), to avoid append block info.
  return data_;
}

void Block::update_data(void* data_new) {
  //update data_, after the swap in completes.
  data_ = data_new;
}


}  // namespace singa
#endif