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

#include "singa/utils/job_manager.h"

#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include "singa/proto/job.pb.h"

using std::string;
using std::vector;

namespace singa {

JobManager::JobManager(const string& host) {
  host_ = host;
}

bool JobManager::Init() {
#ifdef USE_ZOOKEEPER
  if (!zk_.Init(host_, timeout_)) return false;
  if (!zk_.CreateNode(kZKPathSinga.c_str(), nullptr, 0, nullptr))
    return false;
  if (!zk_.CreateNode(kZKPathSys.c_str(), nullptr, 0, nullptr))
    return false;
  if (!zk_.CreateNode(kZKPathJLock.c_str(), nullptr, 0, nullptr))
    return false;
  if (!zk_.CreateNode(kZKPathHostIdx.c_str(), "0", 0, nullptr))
    return false;
  if (!zk_.CreateNode(kZKPathApp.c_str(), nullptr, 0, nullptr))
    return false;
#endif
  return true;
}

bool JobManager::GenerateJobID(int* id) {
#ifdef USE_ZOOKEEPER
  char buf[kZKBufSize];
  string lock = kZKPathJLock + "/lock-";
  if (!zk_.CreateNode(lock.c_str(), nullptr,
                        ZOO_EPHEMERAL | ZOO_SEQUENCE, buf)) {
    return false;
  }
  *id = atoi(buf + strlen(buf) - 10);
#else
  *id = 0;
#endif
  return true;
}

bool JobManager::GenerateHostList(const char* host_file, const char* job_file,
                                  vector<string>* list) {
  int nprocs = 1;
  list->clear();
  // compute required #process from job conf
  if (job_file != nullptr) {
    ClusterProto cluster;
    google::protobuf::TextFormat::ParseFromString(ExtractClusterConf(job_file),
                                                  &cluster);
    int nworker_procs = cluster.nworker_groups() * cluster.nworkers_per_group()
                        / cluster.nworkers_per_procs();
    int nserver_procs = cluster.nserver_groups() * cluster.nservers_per_group()
                        / cluster.nservers_per_procs();
    if (cluster.server_worker_separate())
      nprocs = nworker_procs + nserver_procs;
    else
      nprocs = std::max(nworker_procs, nserver_procs);
  }
#ifdef USE_ZOOKEEPER
  // get available host list from global conf
  std::ifstream hostfile(host_file);
  if (!hostfile.is_open()) {
    LOG(FATAL) << "Cannot open file: " << host_file;
  }
  vector<string> hosts;
  string host;
  while (!hostfile.eof()) {
    getline(hostfile, host);
    if (!host.length() || host[0] == '#') continue;
    hosts.push_back(host);
  }
  if (!hosts.size()) {
    LOG(FATAL) << "Empty host file";
  }
  // read next host index
  char val[kZKBufSize];
  if (!zk_.GetNode(kZKPathHostIdx.c_str(), val)) return false;
  int next = atoi(val);
  // generate host list
  for (int i = 0; i < nprocs; ++i) {
    list->push_back(hosts[(next + i) % hosts.size()]);
  }
  // write next host index
  next = (next + nprocs) % hosts.size();
  snprintf(val, kZKBufSize, "%d", next);
  if (!zk_.UpdateNode(kZKPathHostIdx.c_str(), val)) return false;
#else
  CHECK_EQ(nprocs, 1) << "To run multi-process job, please enable zookeeper";
  list->push_back("localhost");
#endif
  return true;
}

bool JobManager::ListJobProcs(int job, vector<string>* procs) {
  procs->clear();
#ifdef USE_ZOOKEEPER
  string job_path = GetZKJobWorkspace(job);
  // check job path
  if (!zk_.Exist(job_path.c_str())) {
    LOG(ERROR) << "job " << job << " not exists";
    return true;
  }
  string proc_path = job_path + kZKPathJobProc;
  vector<string> vt;
  // check job proc path
  if (!zk_.GetChild(proc_path.c_str(), &vt)) {
    return false;
  }
  char buf[singa::kZKBufSize];
  for (string pname : vt) {
    pname = proc_path + "/" + pname;
    if (!zk_.GetNode(pname.c_str(), buf)) continue;
    std::string proc = "";
    for (int i = 0; buf[i] != '\0'; ++i) {
      if (buf[i] == ':') {
        buf[i] = '\0';
        proc += buf;
      } else if (buf[i] == '|') {
        proc += buf + i;
      }
    }
    procs->push_back(proc);
  }
  if (!procs->size()) LOG(ERROR) << "job " << job << " not exists";
#endif
  return true;
}

bool JobManager::ListJobs(vector<JobInfo>* jobs) {
  jobs->clear();
#ifdef USE_ZOOKEEPER
  vector<string> vt;
  // get all children in app path
  if (!zk_.GetChild(kZKPathApp.c_str(), &vt)) {
    return false;
  }
  std::sort(vt.begin(), vt.end());
  int size = static_cast<int>(vt.size());
  vector<string> procs;
  for (int i = 0; i < size; ++i) {
    string path = kZKPathApp + "/" + vt[i] + kZKPathJobProc;
    if (!zk_.GetChild(path.c_str(), &procs)) continue;
    JobInfo job;
    string jid = vt[i].substr(vt[i].length()-10);
    job.id = atoi(jid.c_str());
    job.procs = procs.size();
    jobs->push_back(job);
    // may need to delete it
    if (!job.procs && (i + kJobsNotRemoved < size))
        CleanPath(kZKPathApp + "/" + vt[i], true);
  }
#else
  LOG(ERROR) << "Not supported without zookeeper";
#endif
  return true;
}

bool JobManager::Remove(int job) {
#ifdef USE_ZOOKEEPER
  string path = GetZKJobWorkspace(job) + kZKPathJobProc;
  if (zk_.Exist(path.c_str())) {
    return CleanPath(path.c_str(), false);
  }
#else
  LOG(ERROR) << "Not supported without zookeeper";
#endif
  return true;
}

bool JobManager::RemoveAllJobs() {
#ifdef USE_ZOOKEEPER
  if (zk_.Exist(kZKPathApp.c_str())) {
    return CleanPath(kZKPathApp.c_str(), false);
  }
#else
  LOG(ERROR) << "Not supported without zookeeper";
#endif
  return true;
}

bool JobManager::CleanUp() {
#ifdef USE_ZOOKEEPER
  if (zk_.Exist(kZKPathSinga.c_str())) {
    return CleanPath(kZKPathSinga.c_str(), true);
  }
#else
  LOG(ERROR) << "Not supported without zookeeper";
#endif
  return true;
}

bool JobManager::CleanPath(const string& path, bool remove) {
#ifdef USE_ZOOKEEPER
  vector<string> child;
  if (!zk_.GetChild(path.c_str(), &child)) return false;
  for (string c : child) {
    if (!CleanPath(path + "/" + c, true)) return false;
  }
  if (remove) return zk_.DeleteNode(path.c_str());
#else
  LOG(ERROR) << "Not supported without zookeeper";
#endif
  return true;
}

// extract cluster configuration part from the job config file
// TODO(wangsh) improve this function to make it robust
string JobManager::ExtractClusterConf(const char* job_file) {
  std::ifstream fin(job_file);
  CHECK(fin.is_open()) << "cannot open job conf file " << job_file;
  string line;
  string cluster;
  bool in_cluster = false;
  while (!fin.eof()) {
    std::getline(fin, line);
    if (in_cluster == false) {
      size_t pos = line.find("cluster");
      if (pos == std::string::npos) continue;
      in_cluster = true;
      line = line.substr(pos);
      cluster = "";
    }
    if (in_cluster == true) {
      cluster += line + "\n";
      if (line.find("}") != std::string::npos)
        in_cluster = false;
    }
  }
  LOG(INFO) << "cluster configure: " << cluster;
  size_t s_pos = cluster.find("{");
  size_t e_pos = cluster.find("}");
  if (s_pos == std::string::npos || e_pos == std::string::npos) {
    LOG(FATAL) << "cannot extract valid cluster configuration in file: "
               << job_file;
  }
  return cluster.substr(s_pos + 1, e_pos - s_pos-1);
}

}  // namespace singa
