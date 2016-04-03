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

#ifndef SINGA_UTILS_CLUSTER_H_
#define SINGA_UTILS_CLUSTER_H_

#include <glog/logging.h>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>
#include "singa/proto/job.pb.h"
#include "singa/proto/singa.pb.h"
#include "singa/utils/cluster_rt.h"
#include "singa/utils/common.h"
#include "singa/utils/singleton.h"

namespace singa {

/**
 * Cluster is a singleton object, which provides cluster configuations,
 * e.g., the topology of the cluster.
 * All IDs start from 0.
 */
class Cluster {
 public:
  // Cluster is a global singleton in a process
  static Cluster* Setup(int job_id, const SingaProto& singaConf,
                        const ClusterProto& clusterConf);
  static Cluster* Get();

  inline int nserver_groups() const { return cluster_.nserver_groups(); }
  inline int nworker_groups() const { return cluster_.nworker_groups(); }
  inline int nworkers_per_group() const { return cluster_.nworkers_per_group();}
  inline int nservers_per_group() const { return cluster_.nservers_per_group();}
  inline int nworkers_per_procs() const { return cluster_.nworkers_per_procs();}
  inline int nservers_per_procs() const { return cluster_.nservers_per_procs();}
  inline int nworker_groups_per_server_group() const {
    if (nserver_groups() == 0 || nservers_per_group() == 0)
      return 1;
    else
      return cluster_.nworker_groups() / cluster_.nserver_groups();
  }
  /**
   * @return true if the calling procs has server threads, otherwise false
   */
  inline bool has_server() const {
    if (server_worker_separate()) {
      CHECK_LT(procs_id_, nprocs_);
      return procs_id_ >= nworker_procs();
    } else {
      return procs_id_ < nserver_procs();
    }
  }
  /**
   * @return true if the calling procs has worker threads.
   */
  inline bool has_worker() const {
    return procs_id_ < nworker_procs();
  }
  /**
   * @return global procs id, which starts from 0.
   */
  inline int procs_id() const { return procs_id_; }
  inline void set_procs_id(int procs_id) { procs_id_ = procs_id; }
  inline bool server_worker_separate() const {
    return cluster_.server_worker_separate();
  }
  inline int nworker_procs() const {
    return nworker_groups() * nworkers_per_group() / nworkers_per_procs();
  }
  inline int nserver_procs() const {
    return nserver_groups() * nservers_per_group() / nservers_per_procs();
  }
  inline int nprocs() const { return nprocs_; }
  /**
   * @return endpoint of the router of a procs with the specified id
   */
  inline std::string endpoint(int procs_id) const {
    CHECK_LT(procs_id, nprocs());
    CHECK_GE(procs_id, 0);
    return cluster_rt_->GetProcHost(procs_id);
  }
  inline std::string workspace() const { return cluster_.workspace(); }
  inline std::string vis_folder() const {
    return cluster_.workspace() + "/visualization";
  }
  inline std::string checkpoint_folder() const {
    return cluster_.workspace() + "/checkpoint";
  }
  /*
  const int stub_timeout() const { return cluster_.stub_timeout(); }
  const int worker_timeout() const { return cluster_.worker_timeout(); }
  const int server_timeout() const { return cluster_.server_timeout(); }
  */
  inline bool share_memory() const { return cluster_.share_memory(); }
  inline int sync_freq() const { return cluster_.sync_freq(); }
  inline int poll_time() const { return cluster_.poll_time(); }
  ClusterRuntime* runtime() const { return cluster_rt_; }

  /**
   * @return logical procs ID
   */
  inline int ProcsIDOf(int group_id, int id, int flag) {
    return procs_ids_.at(Hash(group_id, id, flag));
  }

  /**
   * @param pid, processs ID
   * @param group_size, num of executors in a group
   * @param procs_size, num of executors in a procs
   *
   * @return a vector with 4 integers:
   * [group start, group end), [start executor, end executor)
   */
  const std::vector<int> ExecutorRng(int pid, int group_size, int procs_size);
  /**
   * Register this process.
   *
   * @param pid physical process id get from OS, all other procs ID refers to
   * logical process ID.
   * @param endpoint unique string for other procs to connect
   */
  void Register(int pid, const std::string& endpoint);

 private:
  void Init(int job, const SingaProto& singaConf,
          const ClusterProto& clusterConf);
  void SetupFolders(const ClusterProto &cluster);
  int Hash(int gid, int id, int flag);

  int procs_id_ = -1;
  int nprocs_ = 0;
  // cluster config proto
  ClusterProto cluster_;
  SingaProto singa_;
  ClusterRuntime* cluster_rt_ = nullptr;
  std::unordered_map<int, int> procs_ids_;
};

}  // namespace singa

#endif  // SINGA_UTILS_CLUSTER_H_
