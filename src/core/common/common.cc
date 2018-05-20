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
//TODO(junzhe) ifdef to counter verify
///only include mutable_data() and data()

namespace singa {

void* Block::mutable_data() {
    std::cout<<"mutable_data() "<<this<<' '<<data_<<std::endl;
    initialized_ = true;
    if (ptrDevice_!=nullptr){
      //Append info.
      stringstream strm2;
      strm2<<this;
      string tempStr2 = strm2.str();
      stringstream strm4;
      auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
      strm4<<t2;
      string tempStr4 = strm4.str();
      string temp = "Mutable "+tempStr2+" "+tempStr4;   
      ptrDevice_->AppendInfo(temp);
    }

    if (data_ == nullptr) {
      cout<<"to sleep"<<endl;
      auto tempData_ = ptrDevice_->GetRealGpuPtrInfo(this);
      cout<<"sleepped to get data_ updated"<<endl;
    }
    //data_ = ptrDevice_->GetRealGpuPtrInfo(this);
    //ptrDevice_->SwapOutInfo(this);
    //ptrDevice_->SwapInInfo(this);
    //std::cout<<"data_ vs new ptr "<<data_<<' '<<ptrDevice_->GetRealGpuPtrInfo(this)<<std::endl;

    // std::cout<<"-------NOTE: from common.cc, data_, block_, Device_ "<<data_<<" "<<this<<" "<<ptrDevice_<<std::endl;
    // auto temp = ptrDevice_->GetRealGpuPtrInfo(this);
    // auto temp2 = ptrDevice_->GetRealGpuPtr(this);
    // std::cout<<"=======NOTE:  ptrDevice_->GetRealGpuPtrInfo(this) is here:      "<<temp<<' '<<temp2<<std::endl;

    return static_cast<char*>(data_) + offset_;
  }


const void* Block::data() const {
    CHECK(initialized_) << "Must initialize data before reading it";
    std::cout<<"data() "<<this<<' '<<data_<<std::endl;
    if (ptrDevice_!=nullptr){
      //Append info.
      stringstream strm2;
      strm2<<this;
      string tempStr2 = strm2.str();
      stringstream strm4;
      auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
      strm4<<t2;
      string tempStr4 = strm4.str();
      string temp = "Read "+tempStr2+" "+tempStr4;
      ptrDevice_->AppendInfo(temp);
    }

    if (data_ == nullptr) {
      cout<<"to sleep"<<endl;
      auto tempData_ = ptrDevice_->GetRealGpuPtrInfo(this);
      cout<<"sleepped to get data_ updated"<<endl;
    }
    //test async, with size 13107200.
    // if (size_ ==13107200){
    //     std::cout<<"swap testing: to swap with size 13107200"<<std::endl;
    //     ptrDevice_->SwapOutInfo(this);
    //     std::cout<<"done"<<std::endl;
    // }
    //update the real ptr, not able to assign to data_ as const function
    //void* data_2 = ptrDevice_->GetRealGpuPtrInfo(this);
    //TODO(junzhe) used for testing phase only.
    //ptrDevice_->SwapOutInfo(this);
    //ptrDevice_->SwapInInfo(this);
    //
    //std::cout<<"data_ vs new ptr "<<data_<<' '<<ptrDevice_->GetRealGpuPtrInfo(this)<<std::endl;
    //static_cast<char*>(ptrDevice_->GetRealGpuPtrInfo(this)) + offset_;

    // std::cout<<"-------NOTE: from common.cc, data_, block_, Device_ "<<data_<<" "<<this<<" "<<ptrDevice_<<std::endl;
    // auto temp = ptrDevice_->GetRealGpuPtrInfo(this);
    // auto temp2 = ptrDevice_->GetRealGpuPtr(this);
    // std::cout<<"=======NOTE:  ptrDevice_->GetRealGpuPtrInfo(this) is here:      "<<temp<<' '<<temp2<<std::endl;
    //CHECK_EQ(data_,ptrDevice_->GetRealGpuPtrInfo(this));
    //return static_cast<char*>(ptrDevice_->GetRealGpuPtrInfo(this)) + offset_;
    return static_cast<char*>(data_) + offset_;
  }

void Block::update_data(void* data_new) {
  data_ = data_new;
  std::cout<<"update_data: "<<this<<' '<<data_<<std::endl;
}


}  // namespace singa
#endif