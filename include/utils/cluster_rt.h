#ifndef SINGA_UTILS_CLUSTER_RT_H_
#define SINGA_UTILS_CLUSTER_RT_H_

#include <zookeeper/zookeeper.h>
#include <string>
#include <vector>

namespace singa {

typedef void (*rt_callback)(void *contest);

/**
 * ClusterRuntime is a runtime service that manages dynamic configuration
 * and status of the whole cluster. It mainly provides following services:
 *    1)  Provide running status of each server/worker
 *    2)  Translate process id to (hostname:port)
 */
class ClusterRuntime {
 public:
  virtual ~ClusterRuntime() {}
  /**
   * Initialize the runtime instance
   */
  virtual bool Init() = 0;
  /**
   * register the process, and get a unique process id
   *
   * \return the process id, -1 if failed
   */
  virtual int RegistProc(const std::string& host_addr, int pid) = 0;
  /**
   * translate the process id to host address
   *
   * \return the host and port, "" if no such proc id 
   */
  virtual std::string GetProcHost(int proc_id) = 0;
  /**
   * Server: watch all workers in a server group,
   * will be notified when all workers have left
   */
  virtual bool WatchSGroup(int gid, int sid, rt_callback fn, void* ctx) = 0;
  /**
   * Worker: join a server group (i.e. start to read/update these servers)
   */
  virtual bool JoinSGroup(int gid, int wid, int s_group) = 0;
  /**
   * Worker: leave a server group (i.e. finish its all work)
   */
  virtual bool LeaveSGroup(int gid, int wid, int s_group) = 0;
};

const int kZKBufSize = 100;
// following paths are global
const std::string kZKPathSinga = "/singa";
const std::string kZKPathSys =   "/singa/sys";
const std::string kZKPathJLock = "/singa/sys/job-lock";
const std::string kZKPathApp =   "/singa/app";
const std::string kZKPathJob =   "/singa/app/job-";
// following paths are local under /singa/app/job-X
const std::string kZKPathJobGroup = "/group";
const std::string kZKPathJobProc =  "/proc";
const std::string kZKPathJobPLock = "/proc-lock";

inline std::string GetZKJobWorkspace(int job_id) {
  char buf[kZKBufSize];
  sprintf(buf, "%010d", job_id);
  return kZKPathJob + buf;
}

struct RTCallback {
  rt_callback fn;
  void* ctx;
};

struct JobInfo {
  int id;
  int procs;
  std::string name;
};

class ZKService {
 public:
  static void ChildChanges(zhandle_t* zh, int type, int state,
                           const char *path, void* watcherCtx);
  ~ZKService();
  bool Init(const std::string& host, int timeout);
  bool CreateNode(const char* path, const char* val, int flag, char* output);
  bool DeleteNode(const char* path);
  bool Exist(const char* path);
  bool GetNode(const char* path, char* output);
  bool GetChild(const char* path, std::vector<std::string>* vt);
  bool WGetChild(const char* path, std::vector<std::string>* vt,
                   RTCallback *cb);

 private:
  const int kNumRetry = 10;
  const int kSleepSec = 1;

  static void WatcherGlobal(zhandle_t* zh, int type, int state,
                            const char *path, void* watcherCtx);

  zhandle_t* zkhandle_ = nullptr;
};

class ZKClusterRT : public ClusterRuntime {
 public:
  ZKClusterRT(const std::string& host, int job_id);
  ZKClusterRT(const std::string& host, int job_id, int timeout);
  ~ZKClusterRT() override;

  bool Init() override;
  int RegistProc(const std::string& host_addr, int pid) override;
  std::string GetProcHost(int proc_id) override;
  bool WatchSGroup(int gid, int sid, rt_callback fn, void* ctx) override;
  bool JoinSGroup(int gid, int wid, int s_group) override;
  bool LeaveSGroup(int gid, int wid, int s_group) override;

 private:
  inline std::string groupPath(int gid) {
    return group_path_ + "/sg" + std::to_string(gid);
  }
  inline std::string workerPath(int gid, int wid) {
    return "/g" + std::to_string(gid) + "_w" + std::to_string(wid);
  }
  
  int timeout_ = 30000;
  std::string host_ = "";
  ZKService zk_;
  std::string workspace_ = "";
  std::string group_path_ = "";
  std::string proc_path_ = "";
  std::string proc_lock_path_ = ""; 
  std::vector<RTCallback*> cb_vec_;
};

class JobManager {
 public:
  explicit JobManager(const std::string& host);
  JobManager(const std::string& host, int timeout);

  bool Init();
  int GenerateJobID();
  bool ListJobs(std::vector<JobInfo>* jobs);
  bool ListJobProcs(int job, std::vector<std::string>* procs);
  bool Clean(int job);
  bool Cleanup();

 private:
  const int kJobsNotRemoved = 10;

  bool CleanPath(const std::string& path, bool remove);

  int timeout_ = 30000;
  std::string host_ = "";
  ZKService zk_;
};

}  // namespace singa

#endif  // SINGA_UTILS_CLUSTER_RT_H_
