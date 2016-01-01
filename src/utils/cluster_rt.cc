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

using std::string;
using std::to_string;
using std::vector;

namespace singa {

void ZKService::ChildChanges(zhandle_t *zh, int type, int state,
                               const char *path, void *watcherCtx) {
  // check if already callback
  RTCallback *cb = static_cast<RTCallback*>(watcherCtx);
  if (cb->fn == nullptr) return;
  if (type == ZOO_CHILD_EVENT) {
    struct String_vector child;
    // check the child list and put another watcher
    int ret = zoo_wget_children(zh, path, ChildChanges, watcherCtx, &child);
    if (ret == ZOK) {
      if (child.count == 0) {
        LOG(INFO) << "child.count = 0 in path: " << path;
        // all workers leave, we do callback now
        (*cb->fn)(cb->ctx);
        cb->fn = nullptr;
      }
    } else {
      LOG(FATAL) << "Unhandled ZK error code: " << ret
                 << " (zoo_wget_children " << path << ")";
    }
  } else {
    LOG(FATAL) << "Unhandled callback type code: "<< type;
  }
}

ZKService::~ZKService() {
  // close zookeeper handler
  zookeeper_close(zkhandle_);
}

char zk_cxt[] = "ClusterRuntime";

bool ZKService::Init(const string& host, int timeout) {
  zoo_set_debug_level(ZOO_LOG_LEVEL_ERROR);
  zkhandle_ = zookeeper_init(host.c_str(), WatcherGlobal, timeout, 0,
                             static_cast<void *>(zk_cxt), 0);
  if (zkhandle_ == NULL) {
    LOG(ERROR) << "Error when connecting to zookeeper servers...";
    LOG(ERROR) << "Please ensure zookeeper service is up in host(s):";
    LOG(ERROR) << host.c_str();
    return false;
  }

  return true;
}

bool ZKService::CreateNode(const char* path, const char* val, int flag,
                               char* output) {
  CHECK(zkhandle_) << "zk handler not initialized";
  char buf[kZKBufSize];
  int ret = 0;
  // send the zk request
  for (int i = 0; i < kNumRetry; ++i) {
    ret = zoo_create(zkhandle_, path, val, val == nullptr ? -1 : strlen(val),
                     &ZOO_OPEN_ACL_UNSAFE, flag, buf, kZKBufSize);
    if (ret == ZNONODE) {
      LOG(WARNING) << "zookeeper parent node of " << path
                  << " not exist, retry later";
    } else if (ret == ZCONNECTIONLOSS) {
      LOG(WARNING) << "zookeeper disconnected, retry later";
    } else {
      break;
    }
    sleep(kSleepSec);
  }
  // copy the node name to output
  if (output != nullptr && (ret == ZOK || ret == ZNODEEXISTS)) {
    snprintf(output, kZKBufSize, "%s", buf);
    // use snprintf instead of strcpy
    // strcpy(output, buf);
  }
  if (ret == ZOK) {
    LOG(INFO) << "created zookeeper node " << buf
              << " (" << (val == nullptr ? "NULL" : val) << ")";
    return true;
  } else if (ret == ZNODEEXISTS) {
    LOG(WARNING) << "zookeeper node " << path << " already exists";
    return true;
  } else if (ret == ZCONNECTIONLOSS) {
    LOG(ERROR) << "Cannot connect to zookeeper, "
               << "please ensure it is running properly...\n"
               << "If want to use zookeeper in our thirdparty folder, "
               << "you can start it by:\n"
               << "$ ./bin/zk-service.sh start";
    return false;
  }
  LOG(FATAL) << "Unhandled ZK error code: " << ret
             << " (zoo_create " << path << ")";
  return false;
}

bool ZKService::DeleteNode(const char* path) {
  CHECK(zkhandle_) << "zk handler not initialized";
  int ret = zoo_delete(zkhandle_, path, -1);
  if (ret == ZOK) {
    LOG(INFO) << "deleted zookeeper node " << path;
    return true;
  } else if (ret == ZNONODE) {
    LOG(WARNING) << "try to delete an non-existing zookeeper node " << path;
    return true;
  }
  LOG(FATAL) << "Unhandled ZK error code: " << ret
             << " (zoo_delete " << path << ")";
  return false;
}

bool ZKService::Exist(const char* path) {
  CHECK(zkhandle_) << "zk handler not initialized";
  struct Stat stat;
  int ret = zoo_exists(zkhandle_, path, 0, &stat);
  if (ret == ZOK) return true;
  else if (ret == ZNONODE) return false;
  LOG(WARNING) << "Unhandled ZK error code: " << ret << " (zoo_exists)";
  return false;
}

bool ZKService::UpdateNode(const char* path, const char* val) {
  CHECK(zkhandle_) << "zk handler not initialized";
  // set version = -1, do not check content version
  int ret = zoo_set(zkhandle_, path, val, strlen(val), -1);
  if (ret == ZOK) {
    return true;
  } else if (ret == ZNONODE) {
    LOG(ERROR) << "zk node " << path << " does not exist";
    return false;
  }
  LOG(FATAL) << "Unhandled ZK error code: " << ret
             << " (zoo_get " << path << ")";
  return false;
}

bool ZKService::GetNode(const char* path, char* output) {
  CHECK(zkhandle_) << "zk handler not initialized";
  struct Stat stat;
  int val_len = kZKBufSize;
  int ret = zoo_get(zkhandle_, path, 0, output, &val_len, &stat);
  if (ret == ZOK) {
    output[val_len] = '\0';
    return true;
  } else if (ret == ZNONODE) {
    LOG(ERROR) << "zk node " << path << " does not exist";
    return false;
  }
  LOG(FATAL) << "Unhandled ZK error code: " << ret
             << " (zoo_get " << path << ")";
  return false;
}

bool ZKService::GetChild(const char* path, vector<string>* vt) {
  CHECK(zkhandle_) << "zk handler not initialized";
  struct String_vector child;
  int ret = zoo_get_children(zkhandle_, path, 0, &child);
  if (ret == ZOK) {
    vt->clear();
    for (int i = 0; i < child.count; ++i) vt->push_back(child.data[i]);
    return true;
  }
  LOG(FATAL) << "Unhandled ZK error code: " << ret
             << " (zoo_get_children " << path << ")";
  return false;
}

bool ZKService::WGetChild(const char* path, vector<string>* vt,
                            RTCallback *cb) {
  CHECK(zkhandle_) << "zk handler not initialized";
  struct String_vector child;
  int ret = zoo_wget_children(zkhandle_, path, ChildChanges, cb, &child);
  if (ret == ZOK) {
    vt->clear();
    for (int i = 0; i < child.count; ++i) vt->push_back(child.data[i]);
    return true;
  }
  LOG(FATAL) << "Unhandled ZK error code: " << ret
             << " (zoo_get_children " << path << ")";
  return false;
}


void ZKService::WatcherGlobal(zhandle_t * zh, int type, int state,
                                const char *path, void *watcherCtx) {
  if (type == ZOO_SESSION_EVENT) {
    if (state == ZOO_CONNECTED_STATE)
      LOG(INFO) << "GLOBAL_WATCHER connected to zookeeper successfully!";
    else if (state == ZOO_EXPIRED_SESSION_STATE)
      LOG(INFO) << "GLOBAL_WATCHER zookeeper session expired!";
  }
}

ClusterRuntime::ClusterRuntime(const string& host, int job_id)
    : ClusterRuntime(host, job_id, 30000) {}

ClusterRuntime::ClusterRuntime(const string& host, int job_id, int timeout) {
  host_ = host;
  timeout_ = timeout;
  workspace_ = GetZKJobWorkspace(job_id);
  group_path_ = workspace_ + kZKPathJobGroup;
  proc_path_ = workspace_ + kZKPathJobProc;
  proc_lock_path_ = workspace_ + kZKPathJobPLock;
}

ClusterRuntime::~ClusterRuntime() {
  // release callback vector
  for (RTCallback* p : cb_vec_) {
    delete p;
  }
}

bool ClusterRuntime::Init() {
  if (!zk_.Init(host_, timeout_)) return false;
  if (!zk_.CreateNode(kZKPathSinga.c_str(), nullptr, 0, nullptr))
    return false;
  if (!zk_.CreateNode(kZKPathApp.c_str(), nullptr, 0, nullptr))
    return false;
  if (!zk_.CreateNode(workspace_.c_str(), nullptr, 0, nullptr))
    return false;
  if (!zk_.CreateNode(group_path_.c_str(), nullptr, 0, nullptr))
    return false;
  if (!zk_.CreateNode(proc_path_.c_str(), nullptr, 0, nullptr))
    return false;
  if (!zk_.CreateNode(proc_lock_path_.c_str(), nullptr, 0, nullptr))
    return false;
  return true;
}

int ClusterRuntime::RegistProc(const string& host_addr, int pid) {
  char buf[kZKBufSize];
  string lock = proc_lock_path_ + "/lock-";
  if (!zk_.CreateNode(lock.c_str(), nullptr,
                        ZOO_EPHEMERAL | ZOO_SEQUENCE, buf)) {
    return -1;
  }
  // get all children in lock folder
  vector<string> vt;
  if (!zk_.GetChild(proc_lock_path_.c_str(), &vt)) {
    return -1;
  }
  // find own position among all locks
  int id = -1;
  std::sort(vt.begin(), vt.end());
  for (int i = 0; i < static_cast<int>(vt.size()); ++i) {
    if (proc_lock_path_+"/"+vt[i] == buf) {
      id = i;
      break;
    }
  }
  if (id == -1) {
    LOG(ERROR) << "cannot find own node " << buf;
    return -1;
  }
  // create a new node in proc path
  string path = proc_path_ + "/proc-" + to_string(id);
  string content = host_addr + "|" + to_string(pid);
  if (!zk_.CreateNode(path.c_str(), content.c_str(), ZOO_EPHEMERAL,
                      nullptr)) {
    return -1;
  }
  return id;
}

bool ClusterRuntime::WatchSGroup(int gid, int sid, rt_callback fn, void *ctx) {
  CHECK_NOTNULL(fn);
  string path = groupPath(gid);
  // create zk node
  if (!zk_.CreateNode(path.c_str(), nullptr, 0, nullptr)) return false;
  vector<string> child;
  // store the callback function and context for later usage
  RTCallback *cb = new RTCallback;
  cb->fn = fn;
  cb->ctx = ctx;
  cb_vec_.push_back(cb);
  // start to watch on the zk node, does not care about the first return value
  return zk_.WGetChild(path.c_str(), &child, cb);
}

std::string ClusterRuntime::GetProcHost(int proc_id) {
  char val[kZKBufSize];
  // construct file name
  string path = proc_path_ + "/proc-" + to_string(proc_id);
  if (!zk_.GetNode(path.c_str(), val)) return "";
  int len = strlen(val) - 1;
  while (len && val[len] != '|') --len;
  CHECK(len);
  val[len] = '\0';
  return string(val);
}

bool ClusterRuntime::JoinSGroup(int gid, int wid, int s_group) {
  string path = groupPath(s_group) + workerPath(gid, wid);
  // try to create an ephemeral node under server group path
  return zk_.CreateNode(path.c_str(), nullptr, ZOO_EPHEMERAL, nullptr);
}

bool ClusterRuntime::LeaveSGroup(int gid, int wid, int s_group) {
  string path = groupPath(s_group) + workerPath(gid, wid);
  return zk_.DeleteNode(path.c_str());
}

JobManager::JobManager(const string& host) : JobManager(host, 30000) {}

JobManager::JobManager(const string& host, int timeout) {
  host_ = host;
  timeout_ = timeout;
}

bool JobManager::Init() {
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
  return true;
}

bool JobManager::GenerateJobID(int* id) {
  char buf[kZKBufSize];
  string lock = kZKPathJLock + "/lock-";
  if (!zk_.CreateNode(lock.c_str(), nullptr,
                        ZOO_EPHEMERAL | ZOO_SEQUENCE, buf)) {
    return false;
  }
  *id = atoi(buf + strlen(buf) - 10);
  return true;
}

bool JobManager::GenerateHostList(const char* host_file, const char* job_file,
                                  vector<string>* list) {
  int nprocs = 1;
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
  list->clear();
  for (int i = 0; i < nprocs; ++i) {
    list->push_back(hosts[(next + i) % hosts.size()]);
  }
  // write next host index
  next = (next + nprocs) % hosts.size();
  snprintf(val, kZKBufSize, "%d", next);
  if (!zk_.UpdateNode(kZKPathHostIdx.c_str(), val)) return false;
  return true;
}

bool JobManager::ListJobProcs(int job, vector<string>* procs) {
  procs->clear();
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
  return true;
}

bool JobManager::ListJobs(vector<JobInfo>* jobs) {
  // get all children in app path
  jobs->clear();
  vector<string> vt;
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
  return true;
}

bool JobManager::Remove(int job) {
  string path = GetZKJobWorkspace(job) + kZKPathJobProc;
  if (zk_.Exist(path.c_str())) {
    return CleanPath(path.c_str(), false);
  }
  return true;
}

bool JobManager::RemoveAllJobs() {
  if (zk_.Exist(kZKPathApp.c_str())) {
    return CleanPath(kZKPathApp.c_str(), false);
  }
  return true;
}

bool JobManager::CleanUp() {
  if (zk_.Exist(kZKPathSinga.c_str())) {
    return CleanPath(kZKPathSinga.c_str(), true);
  }
  return true;
}

bool JobManager::CleanPath(const string& path, bool remove) {
  vector<string> child;
  if (!zk_.GetChild(path.c_str(), &child)) return false;
  for (string c : child) {
    if (!CleanPath(path + "/" + c, true)) return false;
  }
  if (remove) return zk_.DeleteNode(path.c_str());
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
