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

#include "singa/utils/channel.h"

#include "singa/utils/logging.h"
#include "singa/utils/singleton.h"

namespace singa {

ChannelManager::~ChannelManager() {
  for (auto it : name2ptr_) {
    if (it.second != nullptr) delete (it.second);
  }
}

void ChannelManager::Init() {
  // do nothing here
}

void ChannelManager::SetDefaultDir(const char* dir) {
  if (dir != nullptr) {
    dir_ = dir;
    if (dir[dir_.length() - 1] != '/') dir_ += '/';
  }
}

Channel* ChannelManager::GetInstance(const std::string& channel) {
  // find the channel
  if (name2ptr_.find(channel) == name2ptr_.end()) {
    // create new channel
    Channel* chn = new Channel(channel);
    chn->SetDestFilePath(dir_ + channel);
    chn->EnableDestFile(true);
    name2ptr_[channel] = chn;
  }
  return name2ptr_[channel];
}

Channel::Channel(const std::string& name) { name_ = name; }

Channel::~Channel() {
  if (os_.is_open()) os_.close();
}

void Channel::SetDestFilePath(const std::string& file) {
  // file is append only
  if (os_.is_open()) os_.close();
  {
    std::ifstream fin(file.c_str());
    if (fin.good())
      LOG(WARNING) << "Messages will be appended to an existed file: " << file;
  }
  os_.open(file.c_str(), std::ios::app);
  if (os_.is_open() == false)
    LOG(WARNING) << "Cannot open channel file (" << file << ")";
}

void Channel::Send(const std::string& message) {
  if (stderr_) fprintf(stderr, "%s\n", message.c_str());
  if (file_ && os_.is_open()) os_ << message << "\n";
  // TODO(wangwei) flush
}

void Channel::Send(const google::protobuf::Message& message) {
  if (stderr_) fprintf(stderr, "%s\n", message.DebugString().c_str());
  if (file_ && os_.is_open()) message.SerializeToOstream(&os_);
  // TODO(wangwei) flush
}

void InitChannel(const char* argv) {
  ChannelManager* mng = Singleton<ChannelManager>().Instance();
  mng->Init();
}

void SetChannelDirectory(const char* path) {
  ChannelManager* mng = Singleton<ChannelManager>().Instance();
  mng->SetDefaultDir(path);
}

Channel* GetChannel(const std::string& channel_name) {
  ChannelManager* mng = Singleton<ChannelManager>().Instance();
  return mng->GetInstance(channel_name);
}

}  // namespace singa
