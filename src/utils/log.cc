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

#include "singa/utils/log.h"
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <cstdio>

namespace singa {

using std::string;

pid_t GetTID() {
  return (pid_t)(uintptr_t)pthread_self();
}

void Display(const std::string& info, const char* file, int line) {
  time_t rw_time = time(nullptr);
  struct tm tm_time;
  localtime_r(&rw_time, &tm_time);
  printf("[%02d%02d %02d:%02d:%02d %5d:%03d %s:%d] %s\n",
         1 + tm_time.tm_mon,
         tm_time.tm_mday,
         tm_time.tm_hour,
         tm_time.tm_min,
         tm_time.tm_sec,
         getpid(),
         static_cast<unsigned>(GetTID())%1000,
         file,
         line,
         info.c_str());
}

}  // namespace singa
