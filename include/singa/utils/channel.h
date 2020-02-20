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

#ifndef SINGA_UTILS_CHANNEL_H_
#define SINGA_UTILS_CHANNEL_H_

#include <google/protobuf/message.h>

#include <fstream>
#include <iostream>
#include <map>
#include <string>

namespace singa {

/// Channel for appending metrics or other information into files or screen.
class Channel {
 public:
  explicit Channel(const std::string& name);
  ~Channel();

  /// Return the channel name, which is also used for naming the output file.
  inline const std::string& GetName() { return name_; }
  /// Disabled by default.
  inline void EnableDestStderr(bool enable) { stderr_ = enable; }
  /// Enabled by default.
  inline void EnableDestFile(bool enable) { file_ = enable; }
  /// Reset the output file path.
  /// The dest file is named as global dir + channel name by default.
  void SetDestFilePath(const std::string& file);
  /// Append a string message
  void Send(const std::string& message);
  /// Append a protobuf message
  void Send(const google::protobuf::Message& message);

 private:
  std::string name_ = "";
  bool stderr_ = false;
  bool file_ = false;
  std::ofstream os_;
};

class ChannelManager {
 public:
  ChannelManager() {}
  ~ChannelManager();

  void Init();
  void SetDefaultDir(const char* dir);
  Channel* GetInstance(const std::string& channel);

 private:
  std::string dir_ = "";
  std::map<std::string, Channel*> name2ptr_;
};

/// Initial function for global usage of channel.
/// 'argv' is for future use.
void InitChannel(const char* argv);
/// Set the directory name for persisting channel content
void SetChannelDirectory(const char* path);
/// Get the channel instance
Channel* GetChannel(const std::string& channel_name);

}  // namespace singa

#endif  // SINGA_UTILS_CHANNEL_H__
