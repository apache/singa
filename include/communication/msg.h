#ifndef INCLUDE_COMMUNICATION_MSG_H_
#define INCLUDE_COMMUNICATION_MSG_H_
#include <string>
#include <czmq.h>
#include <glog/logging.h>

using std::string;
namespace singa {
class BaseMsg{
  public:
  /**
    * Destructor to free memory
    */
  virtual ~BaseMsg(){};
  /**
    * @param first worker/server group id
    * @param id worker/server id within the group
    * @param flag 0 for server, 1 for worker, 2 for stub
    */
  virtual void set_src(int first, int second, int flag)=0;
  virtual void set_dst(int first, int second, int flag)=0;
  virtual void set_src(int procs_id, int flag)=0;
  virtual void set_dst(int procs_id, int flag)=0;
  virtual int src_first() const=0;
  virtual int dst_first() const=0;
  virtual int src_second() const=0;
  virtual int dst_second() const=0;
  virtual int src_flag() const=0;
  virtual int dst_flag() const=0;
  virtual void set_type(int type)=0;
  virtual int type() const=0;
  virtual void set_target(int first, int second)=0;
  virtual int target_first() const=0;
  virtual int target_second() const=0;

  /**
   * Copy src and dst address, including first, id, flag
   */
  virtual BaseMsg* CopyAddr()=0;
  virtual void SetAddr(BaseMsg* msg)=0;

  /**
   * Add a frame (a chunck of bytes) into the message
   */
  virtual void add_frame(const void*, int nBytes)=0;
  virtual int frame_size()=0;
  virtual void* frame_data()=0;
  /**
    * Move the cursor to the next frame
    * @return true if the next frame is not NULL; otherwise false
    */
  virtual bool next_frame()=0;
};
// TODO make it a compiler argument
#define USE_ZMQ

#ifdef USE_ZMQ
class Msg : public BaseMsg{
 public:
  Msg() {
    msg_=zmsg_new();
  }
  virtual ~Msg(){
    if(msg_!=NULL)
      zmsg_destroy(&msg_);
  }
  virtual void set_src(int first, int second, int flag){
    src_=(first<<kOff1)|(second<<kOff2)|flag;
  }
  virtual void set_dst(int first, int second, int flag){
    dst_=(first<<kOff1)|(second<<kOff2)|flag;
  }
  virtual void set_src(int procs_id, int flag){
    set_src(procs_id, 0, flag);
  }
  virtual void set_dst(int procs_id, int flag){
    set_dst(procs_id, 0, flag);
  }
  int src() const {
    return src_;
  }
  int dst() const {
    return dst_;
  }
  virtual int src_first() const {
    int ret=src_>>kOff1;
    return ret;
  }

  virtual int dst_first() const{
    int ret=dst_>>kOff1;
    return ret;
  }
  virtual int src_second() const{
    int ret=(src_&kMask1)>>kOff2;
    return ret;
  }
  virtual int dst_second() const{
    int ret=(dst_&kMask1)>>kOff2;
    return ret;
  }
  virtual int src_flag() const{
    int ret=src_&kMask2;
    return ret;
  }
  virtual int dst_flag() const{
    int ret=dst_&kMask2;
    return ret;
  }

  void SwapAddr(){
    std::swap(src_,dst_);
  }

  virtual void set_type(int type){
    type_=type;
  }
  virtual int type() const{
    return type_;
  }

  virtual void set_target(int first, int second){
    target_first_=first;
    target_second_=second;
  }
  virtual int target_first() const{
    return target_first_;
  }
  virtual int target_second() const{
    return target_second_;
  }

 virtual BaseMsg* CopyAddr(){
    Msg* msg=new Msg();
    msg->src_=src_;
    msg->dst_=dst_;
    return msg;
  }

  virtual void SetAddr(BaseMsg* msg){
    src_=(static_cast<Msg*>(msg))->src_;
    dst_=(static_cast<Msg*>(msg))->dst_;
  }

  virtual void add_frame(const void* addr, int nBytes){
    zmsg_addmem(msg_, addr, nBytes);
  }
  virtual int frame_size(){
    return zframe_size(frame_);
  }

  virtual void* frame_data(){
    return zframe_data(frame_);
  }

  virtual bool next_frame(){
    frame_=zmsg_next(msg_);
    return frame_!=NULL;
  }

  void ParseFromZmsg(zmsg_t* msg){
    char* tmp=zmsg_popstr(msg);
    sscanf(tmp, "%d %d %d %d %d",
        &src_, &dst_, &type_, &target_first_, &target_second_);
    //LOG(ERROR)<<"recv "<<src_<<" "<<dst_<<" "<<target_;
    frame_=zmsg_next(msg);
    msg_=msg;
  }

  zmsg_t* DumpToZmsg(){
    zmsg_pushstrf(msg_, "%d %d %d %d %d",
        src_, dst_, type_, target_first_, target_second_);
    //LOG(ERROR)<<"send "<<src_<<" "<<dst_<<" "<<target_;
    zmsg_t *tmp=msg_;
    msg_=NULL;
    return tmp;
  }

 protected:
  static const unsigned int kOff1=16, kOff2=4;
  static const unsigned int kMask1=(1<<kOff1)-1, kMask2=(1<<kOff2)-1;
  int src_, dst_;
  int type_, target_first_, target_second_;
  zmsg_t* msg_;
  zframe_t *frame_;
};
#endif
inline void DeleteMsg(Msg** msg){
  delete *msg;
  *msg=nullptr;
}


} /* singa */

#endif // INCLUDE_COMMUNICATION_MSG_H_
