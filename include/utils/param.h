#ifndef INCLUDE_UTILS_PARAM_H_
#define INCLUDE_UTILS_PARAM_H_
#include <vector>
#include <string>
#include <map>
#include <functional>
#include "proto/model.pb.h"
#include "utils/blob.h"
#include "communication/msg.h"
// Base paramter class.
namespace singa {
class Param {
 public:
  Param();
  virtual ~Param(){ }
  /**
   * Generate the message for a get request, i.e., get parameters from a server
   *
   * This function is called at worker/stub side.
   * @param copy decides whether to copy the parameter values from the server.
   * @param slice_idx index of the slice from which the message is generated.
   * @return generated message without setting src, dst, target fields.
   */
  virtual Msg* GenGetMsg(bool copy, int slice_idx);
  /**
   * Generate the message for a put request, i.e., put parameters to a server.
   * \copydetails GenGetMsg(bool, int);
   */
  virtual Msg* GenPutMsg(bool copy, int slice_idx);
  /**
   * Generate the message for a update request, i.e., pass info to server for
   * parameter update.
   * \copydetails GenGetMsg(bool, int);
   */
  virtual Msg* GenUpdateMsg(bool copy, int slice_idx);
  /**
   * Generate the message for a synchronization request between server groups.
   *
   * This function is called at server side where the Param is actually a slice
   * of an original Param object.
   * */
  virtual Msg* GenSyncMsg();
  /**
   * Generate the message to response the update request.
   *
   * This function is called at the server side, where the Param is actually a slice
   * of an original Param object.
   * @param copy if true copy the parameter value into the message, otherwise
   * only transfer the pointer of the parameter values.
   * @return response message pointer
   */
  virtual Msg* GenUpdateResponseMsg(bool copy);

  /**
   * Server handling function for get request.
   *
   * @param msg  request message
   * @return resposne message
   */
  virtual Msg* HandleGetMsg(Msg** msg);
  /**
   * Server handling function for put request.
   *
   * \copydetails HandleGetMsg(Msg**)
   */
  virtual Msg* HandlePutMsg(Msg** msg);
  /**
   * Server handling function for synchronization message
   *
   * \copydetails HandleGetMsg(Msg**)
   */
  virtual Msg* HandleSyncMsg(Msg** msg);

<<<<<<< HEAD
  /**
   * Server parses update request message.
   *
   * @param msg
   * @return 1 for copy, 0 for no copy
   */
  virtual int ParseUpdateMsg(Msg** msg);
  /**
   * Worker/Stub parsing function for get response.
   *
   * @param msg
   * @param slice_idx index for the slice
   */
  virtual int ParseGetResponseMsg(Msg** msg, int slice_idx);
  /**
   * Worker/Server parsing function for update response
   *
   * \copydetails ParseGetResponseMsg(Msg**, int);
   */
  virtual int ParseUpdateResponseMsg(Msg** msg, int slice_idx);
  /**
   * Server parsing function for synchronization response.
   *
   * \copydetails ParseGetResponseMsg(Msg** , int);
   */
  virtual int ParseSyncResponseMsg(Msg** msg, int slice_idx);

  /**
   * Setup param object
   *
   * @param proto includes learning rate/weight decay multipliers
   * @param shape
   */
  virtual void Setup(const ParamProto& proto, const std::vector<int>& shape);
  virtual void Setup(const vector<int>& shape);
  /*
   * Fill the values according to initmethod, e.g., gaussian distribution
   *
   * @param version initial version
   */
  virtual void InitValues(int version=0);
  /**
   * Share the data blob from other Param objects.
   *
   * @param other the Param object whose owner owns the data blob
   */
  void ShareData(shared_ptr<Param> other){
    proto_.set_owner(other->owner());
    if(data_!=nullptr)
      CHECK(std::equal(data_->shape().begin(), data_->shape().end(),
          other->data_->shape().begin()));
    data_=other->data_;
  }
  float learning_rate_multiplier() {
    return proto_.learning_rate_multiplier();
  }
  float weight_decay_multiplier() {
    return proto_.weight_decay_multiplier();
  }
  const std::string& name() {
    return proto_.name();
  }
  /**
   * if the Param shares data with others, then owner is the id of that param.
   * otherwise it is itself's id.
   */
  const int owner() const{
    return proto_.owner();
  }
  int id() const{
    return proto_.id();
  }
  void set_id(int id){
    proto_.set_id(id);
    proto_.set_owner(id);
  }

  /**
   * return the version of the parameter value shared by multiple workers
   */
  int version() const {
    return data_->version();
  }

  void set_version(int v) {
    data_->set_version(v); // TODO read version from data blob
  }

  /**
   * return the version of the parameter value local to a worker
   */
  int local_version() const {
    return local_version_;
  }

  void set_local_version(int v){
    local_version_=v;
  }
   /**
    * @return num of floats.
    */
  int size() const {
    return data_->count();
  }
  /**
   * Return const mem address for the content of this parameter
   */
  const Blob<float> &data() {
    return *data_;
  }
  Blob<float> *mutable_data() {
    return data_.get();
  }
  /**
   * Return gradient of this parameter
   */
  const Blob<float> &grad() {
    return grad_;
  }
  Blob<float> *mutable_grad() {
    return &grad_;
  }

  const Blob<float> &history() {
    return history_;
  }
  Blob<float> *mutable_history() {
    return &history_;
  }

  float* mutable_cpu_data(){
    return data_->mutable_cpu_data();
  }
  float* mutable_cpu_grad(){
    return grad_.mutable_cpu_data();
  }
  float* mutable_cpu_history(){
    return history_.mutable_cpu_data();
  }
  int slice_start() const {
    return slice_start_;
  }

  int num_slices() const {
    return num_slices_;
  }

  void AddSlice(int slice_id, int size);

 protected:
  void ParseResponseMsg(Msg** msg, int slice_idx);

 protected:

  /**
   * name of the parameter used to share wights between neuralnets
   */
  std::string name_;
  shared_ptr<Blob<float>> data_;
  int slice_start_, num_slices_;
  vector<int> slice_offset_, slice_size_;
  vector<bool> pending_put_,pending_get_, pending_update_;
  int num_pending_requests_;
  //! gradient, history gradient of this parameter
  Blob<float> grad_, history_;
  ParamProto proto_;
  int local_version_;
};
}  // namespace singa

#endif  // INCLUDE_UTILS_PARAM_H_
