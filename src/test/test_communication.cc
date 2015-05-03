#include <thread>
#include <vector>
#include "gtest/gtest.h"
#include "communication/msg.h"
#include "communication/socket.h"
using std::vector;
using namespace singa;

const char* ping="PING",*pong="PONG";
/**
 * Connect dealer with (gid, id, flag) to stub router
 */
void Connect(Dealer* dealer, int gid, int id, int flag){
  dealer->Connect("inproc://router");
  Msg msg;
  msg.set_src(gid, id, flag);
  msg.set_dst(0,0,2);
  msg.set_type(0);
  msg.add_frame(ping, 4);
  dealer->Send(&msg);
}

/**
 * Dealer thread, ping-pong with the stub router
 */
void DealerPingPong(int id){
  Dealer* dealer=new Dealer();
  Connect(dealer, 0, id, 0);
  Msg* msg=dealer->Receive();
  int flag=msg->src_flag();
  ASSERT_EQ(2, flag);
  ASSERT_EQ(0, msg->dst_group_id());
  ASSERT_EQ(id, msg->dst_id());
  ASSERT_STREQ(pong, (char*)msg->frame_data());
  delete msg;
  delete dealer;
}

/**
 * Worker thread, connect to router and communicate with server thread
 */
void WorkerDealer(int sid, int did){
  Dealer* dealer=new Dealer();
  Connect(dealer, 0, sid, 0);
  for(int i=0;i<2;i++){
    {
      Msg msg;
      msg.set_src(0, sid, 0);
      msg.set_dst(0, did, 1);
      msg.set_type(3);
      msg.set_target(i);
      dealer->Send(&msg);
    }
    {
      Msg *msg=dealer->Receive();
      ASSERT_EQ(0, msg->src_group_id());
      ASSERT_EQ(did, msg->src_id());
      ASSERT_EQ(1, msg->src_flag());
      delete msg;
    }
  }
  delete dealer;
}

/**
 * Server thread, connect to router and communicate with worker thread
 */
void ServerDealer(int id, int n){
  Dealer* dealer=new Dealer();
  Connect(dealer, 0, id, 1);
  for(int i=0;i<n;i++){
    Msg *msg=dealer->Receive();
    Msg reply;
    reply.set_dst(msg->src_group_id(), msg->src_id(), msg->src_flag());
    reply.set_src(0, id, 1);
    dealer->Send(&reply);
    delete msg;
  }
  delete dealer;
}

TEST(CommunicationTest, DealerRouterPingPong){
  int n=2;
  vector<std::thread> threads;
  for(int i=0;i<n;i++)
    threads.push_back(std::thread(DealerPingPong, i));
  Router* router=new Router();
  router->Bind("");
  for(int k=0;k<n;k++){
    Msg* msg=router->Receive();
    ASSERT_EQ(0, msg->src_group_id());
    ASSERT_EQ(2, msg->dst_flag());
    ASSERT_STREQ(ping, (char*)msg->frame_data());

    Msg reply;
    reply.set_src(0,0,2);
    reply.set_dst(msg->src_group_id(), msg->src_id(), msg->src_flag());
    reply.add_frame(pong, 4);
    router->Send(&reply);
    delete msg;
  }

  delete router;
  for(auto& thread:threads)
    thread.join();
}

TEST(CommunicationTest, nWorkers1Server){
  int nworker=2;
  vector<std::thread> threads;
  for(int i=0;i<nworker;i++)
    threads.push_back(std::thread(WorkerDealer, i, 0));
  //threads.push_back(std::thread(ServerDealer, 0, 4));
  Router* router=new Router();
  router->Bind("");
  int nmsg=4*nworker;
  int k=0;
  while(nmsg>0){
    Msg* msg=router->Receive();
    if(2== msg->dst_flag()){
      ASSERT_STREQ(ping, (char*)msg->frame_data());
      k++;
      if(k==nworker)
        threads.push_back(std::thread(ServerDealer, 0, 2*nworker));
    }else{
      nmsg--;
      router->Send(msg);
    }
    delete msg;
  }
  delete router;
  for(auto& thread:threads)
    thread.join();
}

TEST(CommunicationTest, 2Workers2Server){
  vector<std::thread> threads;
  threads.push_back(std::thread(WorkerDealer, 0, 0));
  threads.push_back(std::thread(WorkerDealer, 1, 1));
  threads.push_back(std::thread(ServerDealer, 0, 2));
  threads.push_back(std::thread(ServerDealer, 1, 2));
  Router* router=new Router();
  router->Bind("");
  int n=8;
  while(n>0){
    Msg* msg=router->Receive();
    if(2== msg->dst_flag()){
      ASSERT_STREQ(ping, (char*)msg->frame_data());
    }else{
      n--;
      router->Send(msg);
    }
    delete msg;
  }
  delete router;
  for(auto& thread:threads)
    thread.join();
}
