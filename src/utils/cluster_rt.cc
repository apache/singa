#include "utils/cluster_rt.h"

#include <glog/logging.h>
#include <algorithm>

using std::string;
using std::to_string;

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
      LOG(ERROR) << "Unhandled ZK error code: " << ret
                 << " (zoo_wget_children)";
    }
  } else {
    LOG(ERROR) << "Unhandled callback type code: "<< type;
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
  // copy the node name ot output
  if (output != nullptr && (ret == ZOK || ret == ZNODEEXISTS)) {
    strcpy(output, buf);
  }
  if (ret == ZOK) {
    LOG(INFO) << "created zookeeper node " << buf
              << " (" << (val == nullptr ? "NULL" : val) << ")";
    return true;
  } else if (ret == ZNODEEXISTS) {
    LOG(WARNING) << "zookeeper node " << path << " already exists";
    return true;
  }
  LOG(ERROR) << "Unhandled ZK error code: " << ret << " (zoo_create)";
  return false;
}

bool ZKService::DeleteNode(const char* path) {
  int ret = zoo_delete(zkhandle_, path, -1);
  if (ret == ZOK) {
    LOG(INFO) << "deleted zookeeper node " << path;
    return true;
  } else if (ret == ZNONODE) {
    LOG(WARNING) << "try to delete an non-existing zookeeper node " << path;
    return true;
  }
  LOG(ERROR) << "Unhandled ZK error code: " << ret << " (zoo_delete)";
  return false;
}

bool ZKService::Exist(const char* path) {
  struct Stat stat;
  int ret = zoo_exists(zkhandle_, path, 0, &stat);
  if (ret == ZOK) return true;
  else if (ret == ZNONODE) return false;
  LOG(ERROR) << "Unhandled ZK error code: " << ret << " (zoo_exists)";
  return false;
}

bool ZKService::GetNode(const char* path, char* output) {
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
  LOG(ERROR) << "Unhandled ZK error code: " << ret << " (zoo_get)";
  return false;
}

bool ZKService::GetChild(const char* path, std::vector<string>* vt) {
  struct String_vector child;
  int ret = zoo_get_children(zkhandle_, path, 0, &child);
  if (ret == ZOK) {
    for (int i = 0; i < child.count; ++i) vt->push_back(child.data[i]);
    return true;
  }
  LOG(ERROR) << "Unhandled ZK error code: " << ret << " (zoo_get_children)";
  return false;
}

bool ZKService::WGetChild(const char* path, std::vector<std::string>* vt,
                            RTCallback *cb) {
  struct String_vector child;
  int ret = zoo_wget_children(zkhandle_, path, ChildChanges, cb, &child);
  if (ret == ZOK) {
    for (int i = 0; i < child.count; ++i) vt->push_back(child.data[i]);
    return true;
  }
  LOG(ERROR) << "Unhandled ZK error code: " << ret << " (zoo_get_children)";
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

ZKClusterRT::ZKClusterRT(const string& host) : ZKClusterRT(host, 30000) {}

ZKClusterRT::ZKClusterRT(const string& host, int timeout) {
  host_ = host;
  timeout_ = timeout;
}

ZKClusterRT::~ZKClusterRT() {
  // release callback vector
  for (RTCallback* p : cb_vec_) {
    delete p;
  }
}

bool ZKClusterRT::Init() {
  if (!zk_.Init(host_, timeout_)) return false;
  // create kZKPathSinga
  if (!zk_.CreateNode(kZKPathSinga.c_str(), nullptr, 0, nullptr))
    return false;
  // create kZKPathStatus
  if (!zk_.CreateNode(kZKPathStatus.c_str(), nullptr, 0, nullptr))
    return false;
  // create kZKPathRegist
  if (!zk_.CreateNode(kZKPathRegist.c_str(), nullptr, 0, nullptr))
    return false;
  // create kZKPathRegistProc
  if (!zk_.CreateNode(kZKPathRegistProc.c_str(), nullptr, 0, nullptr))
    return false;
  // create kZKPathRegistLock
  if (!zk_.CreateNode(kZKPathRegistLock.c_str(), nullptr, 0, nullptr))
    return false;
  return true;
}

int ZKClusterRT::RegistProc(const string& host_addr) {
  char buf[kZKBufSize];
  string lock = kZKPathRegistLock+"/lock-";
  if (!zk_.CreateNode(lock.c_str(), nullptr,
                        ZOO_EPHEMERAL | ZOO_SEQUENCE, buf)) {
    return -1;
  }
  // get all children in lock folder
  std::vector<string> vt;
  if (!zk_.GetChild(kZKPathRegistLock.c_str(), &vt)) {
    return -1;
  }
  // find own position among all locks
  int id = -1;
  std::sort(vt.begin(), vt.end());
  for (int i = 0; i < static_cast<int>(vt.size()); ++i) {
    if (kZKPathRegistLock+"/"+vt[i] == buf) {
      id = i;
      break;
    }
  }
  if (id == -1) {
    LOG(ERROR) << "cannot find own node " << buf;
    return -1;
  }
  // create a new node in proc path
  string path = kZKPathRegistProc+"/proc-"+to_string(id);
  if (!zk_.CreateNode(path.c_str(), host_addr.c_str(), ZOO_EPHEMERAL,
                      nullptr)) {
    return -1;
  }
  return id;
}

bool ZKClusterRT::WatchSGroup(int gid, int sid, rt_callback fn, void *ctx) {
  CHECK_NOTNULL(fn);
  string path = groupPath(gid);
  // create zk node
  if (!zk_.CreateNode(path.c_str(), nullptr, 0, nullptr)) return false;
  std::vector<string> child;
  // store the callback function and context for later usage
  RTCallback *cb = new RTCallback;
  cb->fn = fn;
  cb->ctx = ctx;
  cb_vec_.push_back(cb);
  // start to watch on the zk node, does not care about the first return value
  return zk_.WGetChild(path.c_str(), &child, cb);
}

string ZKClusterRT::GetProcHost(int proc_id) {
  // char buf[kZKBufSize];
  char val[kZKBufSize];
  // construct file name
  string path = kZKPathRegistProc+"/proc-"+to_string(proc_id);
  if (!zk_.GetNode(path.c_str(), val)) return "";
  return string(val);
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

JobManager::JobManager(const string& host) : JobManager(host, 30000) {}

JobManager::JobManager(const string& host, int timeout) {
  host_ = host;
  timeout_ = timeout;
}

bool JobManager::Init() {
  return zk_.Init(host_, timeout_);
}

bool JobManager::Clean() {
  if (zk_.Exist(kZKPathSinga.c_str())) {
    return CleanPath(kZKPathSinga.c_str());
  }
  return true;
}

bool JobManager::CleanPath(const std::string& path) {
  std::vector<string> child;
  if (!zk_.GetChild(path.c_str(), &child)) return false;
  for (string c : child) {
    if (!CleanPath(path + "/" + c)) return false;
  }
  return zk_.DeleteNode(path.c_str());
}

}  // namespace singa
