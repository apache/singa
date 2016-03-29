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

#include "singa/utils/cluster_rt.h"

#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include "singa/proto/job.pb.h"

#ifdef USE_ZOOKEEPER
#include "singa/utils/zk_service.h"
#endif

using std::string;
using std::to_string;
using std::vector;

namespace singa {

ClusterRuntime* ClusterRuntime::Create(const std::string&host, int job_id) {
#ifdef USE_ZOOKEEPER
  return new ZKClusterRT(host, job_id);
#else
  return new SPClusterRT();
#endif
}

SPClusterRT::~SPClusterRT() {
  // release callback vector
  for (auto list : grp_callbacks_)
    for (RTCallback* p : list.second) {
    delete p;
  }
}

bool SPClusterRT::Init() {
  return true;
}

int SPClusterRT::RegistProc(const string& host_addr, int pid) {
  int ret;
  lock_.lock();
  proc_list_.push_back(host_addr + std::to_string(pid));
  ret = proc_list_.size()-1;
  lock_.unlock();
  return ret;
}

string SPClusterRT::GetProcHost(int proc_id) {
  if (proc_list_.size() < (unsigned)proc_id + 1) return "";
  return proc_list_[proc_id];
}

bool SPClusterRT::WatchSGroup(int gid, int sid, rt_callback fn, void* ctx) {
  // store the callback function and context for later usage
  RTCallback *cb = new RTCallback;
  cb->fn = fn;
  cb->ctx = ctx;
  lock_.lock();
  if (grp_callbacks_.count(gid) == 0)
    grp_callbacks_[gid] = vector<RTCallback*>{};
  grp_callbacks_[gid].push_back(cb);
  lock_.unlock();
  return true;
}

bool SPClusterRT::JoinSGroup(int gid, int wid, int s_group) {
  lock_.lock();
  if (grp_count_.count(gid) == 0)
    grp_count_[gid] = 0;
  grp_count_[gid]++;
  lock_.unlock();
  return true;
}

bool SPClusterRT::LeaveSGroup(int gid, int wid, int s_group) {
  lock_.lock();
  if (--grp_count_[gid] == 0) {
      for (RTCallback* cb : grp_callbacks_[gid]) {
        (*cb->fn)(cb->ctx);
        cb->fn = nullptr;
      }
  }
  lock_.unlock();
  return true;
}

}  // namespace singa
