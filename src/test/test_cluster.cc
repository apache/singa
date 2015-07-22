#include "gtest/gtest.h"
#include "utils/cluster.h"

using namespace singa;

std::string host = "localhost:2181";

void zk_cb(void *contest) {
  LOG(INFO) << "zk callback: " << static_cast<char *>(contest);
}

TEST(CluserRuntimeTest, GroupManagement) {
  ClusterRuntime* rt = new ZKClusterRT(host);
  ASSERT_EQ(rt->Init(), true);
  ASSERT_EQ(rt->WatchSGroup(1, 1, zk_cb, "test call back"), true);
  ASSERT_EQ(rt->JoinSGroup(1, 1, 1), true);
  ASSERT_EQ(rt->JoinSGroup(1, 2, 1), true);
  ASSERT_EQ(rt->LeaveSGroup(1, 2, 1), true);
  ASSERT_EQ(rt->LeaveSGroup(1, 1, 1), true);
  sleep(3);
  delete rt;
}

TEST(CluserRuntimeTest, ProcessManagement) {
  ClusterRuntime* rt = new ZKClusterRT(host);
  ASSERT_EQ(rt->Init(), true);
  ASSERT_EQ(rt->RegistProc("1.2.3.4:5"), 0);
  ASSERT_EQ(rt->RegistProc("1.2.3.4:6"), 1);
  ASSERT_EQ(rt->RegistProc("1.2.3.4:7"), 2);
  ASSERT_NE(rt->GetProcHost(0), "");
  ASSERT_NE(rt->GetProcHost(1), "");
  ASSERT_NE(rt->GetProcHost(2), "");
  sleep(3);
  delete rt;
}

/**
ClusterProto GenClusterProto(){
  ClusterProto proto;
  int nworker=6, nserver=4;
  proto.set_nworkers(nworker);
  proto.set_nservers(nserver);
  proto.set_nworkers_per_group(3);
  proto.set_nservers_per_group(2);
  proto.set_nthreads_per_worker(1);
  proto.set_nthreads_per_server(2);

  proto.set_hostfile(folder+"/hostfile");

  std::ofstream fout(folder+"/hostfile", std::ofstream::out);
  for(int i=0;i<nworker+nserver;i++){
    char tmp[20];
    sprintf(tmp, "awan-0-%02d-0", i);
    fout<<tmp<<std::endl;
  }
  fout.flush();
  fout.close();
  return proto;
}

TEST(ClusterTest, NoServer){
  ClusterProto proto=GenClusterProto();
  proto.set_nservers(0);
  auto cluster=Cluster::Get(proto, 0);
  ASSERT_EQ(proto.nworkers(),cluster->nworkers());
  ASSERT_EQ(0, cluster->nservers());
  ASSERT_EQ(proto.nworkers_per_group(),cluster->nworkers_per_group());
  ASSERT_EQ(proto.nservers_per_group(),cluster->nservers_per_group());
  ASSERT_FALSE(cluster->AmIServer());
  ASSERT_TRUE(cluster->AmIWorker());
  ASSERT_EQ(0,cluster->group_procs_id());
  ASSERT_EQ(0,cluster->group_id());
  ASSERT_EQ(2, cluster->nworker_groups());
  ASSERT_EQ(0, cluster->nserver_groups());
  ASSERT_STREQ("awan-0-00-0", cluster->host_addr().c_str());

  cluster=Cluster::Get(proto, 5);
  ASSERT_EQ(2,cluster->group_procs_id());
  ASSERT_EQ(1,cluster->group_id());
  ASSERT_EQ(2, cluster->nworker_groups());
  ASSERT_EQ(0, cluster->nserver_groups());
  ASSERT_STREQ("awan-0-05-0", cluster->host_addr().c_str());
}

TEST(ClusterTest, SingleServerGroup){
  ClusterProto proto=GenClusterProto();
  proto.set_nservers(2);
  auto cluster=Cluster::Get(proto, 3);
  ASSERT_FALSE(cluster->AmIServer());
  ASSERT_TRUE(cluster->AmIWorker());
  ASSERT_EQ(0,cluster->group_procs_id());
  ASSERT_EQ(1,cluster->group_id());
  ASSERT_EQ(2, cluster->nworker_groups());
  ASSERT_EQ(1, cluster->nserver_groups());
  ASSERT_STREQ("awan-0-03-0", cluster->host_addr().c_str());

  cluster=Cluster::Get(proto, 7);
  ASSERT_EQ(1,cluster->group_procs_id());
  ASSERT_EQ(0,cluster->group_id());
  ASSERT_EQ(2, cluster->nworker_groups());
  ASSERT_EQ(1, cluster->nserver_groups());
  ASSERT_STREQ("awan-0-07-0", cluster->host_addr().c_str());
}

TEST(ClusterTest, MultiServerGroups){
  ClusterProto proto=GenClusterProto();
  auto cluster=Cluster::Get(proto, 7);
  ASSERT_EQ(1,cluster->group_procs_id());
  ASSERT_EQ(0,cluster->group_id());
  ASSERT_EQ(2, cluster->nworker_groups());
  ASSERT_EQ(2, cluster->nserver_groups());
  ASSERT_STREQ("awan-0-07-0", cluster->host_addr().c_str());

  cluster=Cluster::Get(proto, 8);
  ASSERT_TRUE(cluster->AmIServer());
  ASSERT_FALSE(cluster->AmIWorker());
  ASSERT_EQ(0,cluster->group_procs_id());
  ASSERT_EQ(1,cluster->group_id());
  ASSERT_EQ(2, cluster->nworker_groups());
  ASSERT_EQ(2, cluster->nserver_groups());
  ASSERT_STREQ("awan-0-08-0", cluster->host_addr().c_str());
}
**/
