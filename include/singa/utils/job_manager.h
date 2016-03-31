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

#ifndef SINGA_UTILS_JOB_MANAGER_H_
#define SINGA_UTILS_JOB_MANAGER_H_

#include <string>
#include <vector>

#ifdef USE_ZOOKEEPER
#include "singa/utils/zk_service.h"
#endif

namespace singa {

struct JobInfo {
  int id;
  int procs;
  std::string name;
};

class JobManager {
 public:
  // host is comma separated host:port pairs, each corresponding to a zk server.
  // e.g. "127.0.0.1:3000,127.0.0.1:3001,127.0.0.1:3002"
  explicit JobManager(const std::string& host);

  // NOTICE: Init must be called once, before start to use other functions
  bool Init();
  // generate a unique job id
  bool GenerateJobID(int* id);
  // generate a list of hosts for a job conf
  bool GenerateHostList(const char* host_file, const char* job_file,
                        std::vector<std::string>* list);
  // list all jobs recorded in zk
  bool ListJobs(std::vector<JobInfo>* jobs);
  // list running processes for a job
  bool ListJobProcs(int job, std::vector<std::string>* procs);
  // remove a job path in zk
  bool Remove(int job);
  // remove all job paths in zk
  bool RemoveAllJobs();
  // remove all singa related paths in zk
  bool CleanUp();

 private:
  const int kJobsNotRemoved = 10;

  bool CleanPath(const std::string& path, bool remove);
  std::string ExtractClusterConf(const char* job_file);

  std::string host_ = "";
#ifdef USE_ZOOKEEPER
  int timeout_ = 30000;
  ZKService zk_;
#endif
};

}  // namespace singa

#endif  // SINGA_UTILS_JOB_MANAGER_H_
