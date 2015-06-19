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
  virtual int RegistProc(const std::string& host_addr) = 0;
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

class ZKClusterRT : public ClusterRuntime {
 public:
  explicit ZKClusterRT(const std::string& host);
  ZKClusterRT(const std::string& host, int timeout);
  ~ZKClusterRT() override;

  bool Init() override;
  int RegistProc(const std::string& host_addr) override;
  std::string GetProcHost(int proc_id) override;
  bool WatchSGroup(int gid, int sid, rt_callback fn, void* ctx) override;
  bool JoinSGroup(int gid, int wid, int s_group) override;
  bool LeaveSGroup(int gid, int wid, int s_group) override;

 private:
  struct RTCallback {
    rt_callback fn;
    void* ctx;
  };

  const int kMaxBufLen = 50;
  const int kNumRetry = 10;
  const int kSleepSec = 1;
  const std::string kZPathSinga = "/singa";
  const std::string kZPathStatus = "/singa/status";
  const std::string kZPathRegist = "/singa/regist";
  const std::string kZPathRegistProc = "/singa/regist/proc";
  const std::string kZPathRegistLock = "/singa/regist/lock";

  static void WatcherGlobal(zhandle_t* zh, int type, int state,
                            const char *path, void* watcherCtx);
  static void ChildChanges(zhandle_t* zh, int type, int state,
                           const char *path, void* watcherCtx);
  bool CreateZKNode(const char* path, const char* val, int flag, char* output);
  bool DeleteZKNode(const char* path);
  bool GetZKNode(const char* path, char* output);
  bool GetZKChild(const char* path, std::vector<std::string>* vt);
  inline std::string groupPath(int gid);
  std::string workerPath(int gid, int wid);

  int timeout_ = 30000;
  std::string host_ = "";
  zhandle_t* zkhandle_ = nullptr;
  std::vector<RTCallback*> cb_vec_;
};

}  // namespace singa

#endif  // SINGA_UTILS_CLUSTER_RT_H_
