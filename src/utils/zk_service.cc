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

#include "singa/utils/zk_service.h"

#include <glog/logging.h>
#include <algorithm>

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

char zk_cxt[] = "ZKClusterRT";

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

ZKClusterRT::ZKClusterRT(const string& host, int job_id) {
  host_ = host;
  workspace_ = GetZKJobWorkspace(job_id);
  group_path_ = workspace_ + kZKPathJobGroup;
  proc_path_ = workspace_ + kZKPathJobProc;
  proc_lock_path_ = workspace_ + kZKPathJobPLock;
}

ZKClusterRT::~ZKClusterRT() {
  // release callback vector
  for (RTCallback* p : cb_vec_) {
    delete p;
  }
}

bool ZKClusterRT::Init() {
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

int ZKClusterRT::RegistProc(const string& host_addr, int pid) {
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

std::string ZKClusterRT::GetProcHost(int proc_id) {
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

bool ZKClusterRT::WatchSGroup(int gid, int sid, rt_callback fn, void *ctx) {
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

bool ZKClusterRT::JoinSGroup(int gid, int wid, int s_group) {
  string path = groupPath(s_group) + workerPath(gid, wid);
  // try to create an ephemeral node under server group path
  return zk_.CreateNode(path.c_str(), nullptr, ZOO_EPHEMERAL, nullptr);
}

bool ZKClusterRT::LeaveSGroup(int gid, int wid, int s_group) {
  string path = groupPath(s_group) + workerPath(gid, wid);
  return zk_.DeleteNode(path.c_str());
}

}  // namespace singa
