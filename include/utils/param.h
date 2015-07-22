#ifndef INCLUDE_UTILS_PARAM_H_
#define INCLUDE_UTILS_PARAM_H_
#include <vector>
#include <string>
#include "proto/job.pb.h"
#include "utils/blob.h"
#include "communication/msg.h"

/**
 * Base paramter class.
 *
 * The Param object is a set of parameters, e.g., the (sub) weight matrix or
 * (sub) bias vector.
 *
 * It has at a gradient Blob and data Blob for gradients and parameter values.
 * Since some layers (or neuralnet) share parameter values, the data Blob is a
 * shared pointer which can be assigned to many Param objects' data field.
 *
 * It provides access methods like data(), grad(). It also provides functions
 * for generating messages and parsing messages to transferring the Param
 * objects among worker-worker, worker-server and server-server.
 *
 * Param objects are of different sizes, which makes it hard to acheive
 * load-balance among servers. Hence, we slice large Param objects into small
 * pieces. At the server side, one slice is a Param object.
 */
namespace singa {
class Param {
 public:
  Param();
  virtual ~Param(){ }
  /**
   * Setup param object
   *
   * @param conf param configuration, include learning rate multiplier etc.
   * @param shape one value per dimension
   */
  virtual void Setup(const ParamProto& conf, const std::vector<int>& shape);
  /*
   * Fill the values according to init method, e.g., gaussian distribution.
   *
   * @param version initial version
   */
  virtual void InitValues(int version=0);
  /**
   * Share the data blob from other Param objects.
   *
   * @param other the Param object whose owner owns the data blob
   */
  void ShareFrom(const Param& other);

  /**
   * Scale the learning rate when updating parameters in the Param object
   */
  float learning_rate_multiplier() {
    return proto_.learning_rate_multiplier();
  }
  /**
   * Scale the weight decay when updating parameters in the Param object
   */
  float weight_decay_multiplier() {
    return proto_.weight_decay_multiplier();
  }
  /**
   * Parameter name used for Param re-use in other model or sharing between
   * layers
   */
  const std::string& name() {
    return proto_.name();
  }
  void set_name(const std::string& name) {
    proto_.set_name(name);
  }
  /**
   * If it shares data from others, then owner is the id of that Param,
   * otherwise it is itself's id.
   */
  const int owner() const {
    return proto_.owner();
  }
  /**
   * ID start from 0 and ordered for all Param from the same neuralnet
   */
  int id() const {
    return proto_.id();
  }
  /**
   * Set ID
   */
  void set_id(int id) {
    proto_.set_id(id);
    proto_.set_owner(id);
  }

  /**
   * Param version is stored inside the data blob to enable all Param objs
   * sharing the same values have the same version.
   * @return the param version
   */
  int version() const {
    return data_->version();
  }

  void set_version(int v) {
    data_->set_version(v);
  }

  /**
   * @return the version of the parameter value local to a worker
   */
  int local_version() const {
    return local_version_;
  }

  void set_local_version(int v) {
    local_version_=v;
  }
  const std::string& share_from() const {
    return proto_.share_from();
  }
   /**
    * @return num of floats.
    */
  int size() const {
    return data_->count();
  }
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
  float* mutable_cpu_data() {
    return data_->mutable_cpu_data();
  }
  float* mutable_cpu_grad() {
    return grad_.mutable_cpu_data();
  }
  float* mutable_cpu_history() {
    return history_.mutable_cpu_data();
  }

  /**
   * @return slice start ID
   */
  int slice_start() const {
    return slice_start_;
  }

  int num_slices() const {
    return num_slices_;
  }

  /**
   * Add a slice
   *
   * @param slice_id
   * @param size num of floats for this slice
   */
  void AddSlice(int slice_id, int size);
  /**
   * Init param values from checkpoint blob.
   */
  void FromProto(const BlobProto& blob);
  /**
   * Dump param values to blob.
   */
  void ToProto(BlobProto* blob);
  /**********************Msg related functions***************************/

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
  virtual Msg* GenSyncMsg(int offset, int size);
  /**
   * Generate the messages to response the update requests.
   *
   * This function is called at the server side, where the Param is actually a
   * slice of an original Param object.
   *
   * @param msgs for synchronous training, there would be multiple procs in
   * which workers sharing the same Param (slice) objects. Their update requests
   * is bufferred and handled together. For asynchrnous training, there is only
   * request in msgs.
   * @return response messages
   */
  virtual const vector<Msg*> GenUpdateResponseMsgs(const vector<Msg*>& msgs);

  /**
   * Server handling function for get request.
   *
   * @param msg request
   * @param reserve if true reserve the msg space for the calling function;
   * otherwise the msg should be freed inside the function.
   * @return resposne message
   */
  virtual Msg* HandleGetMsg(Msg** msg, bool reserve = false);
  /**
   * Server handling function for put request.
   *
   * \copydetails HandleGetMsg(Msg**, bool reserve)
   */
  virtual Msg* HandlePutMsg(Msg** msg, bool reserve = false);
  /**
   * Server handling function for synchronization message
   *
   * \copydetails HandleGetMsg(Msg**, bool reserve)
   */
  virtual Msg* HandleSyncMsg(Msg** msg, bool reserve = false);
  /**
   * Worker/Stub parsing function for get response.
   *
   * @param msg
   * @param slice_idx index for the slice
   */
  virtual int ParseGetResponseMsg(Msg* msg, int slice_idx);
  /**
   * Worker/Server parsing function for update response
   *
   * \copydetails ParseGetResponseMsg(Msg**, int);
   */
  virtual int ParseUpdateResponseMsg(Msg* msg, int slice_idx);
  /**
   * Server parse update requests.
   * \copydetails GenUpdateResponseMsgs(const vector<Msg*>& msgs);
   */
  virtual void ParseUpdateMsgs(const vector<Msg*>& msgs);
  /**
   * Server parsing function for synchronization response.
   *
   * \copydetails ParseGetResponseMsg(Msg** , int);
   */
  virtual int ParseSyncResponseMsg(Msg* msg, int slice_idx);

 protected:
  /**
   * Implement the common code of ParseGetResponseMsg and ParseUpdateResponseMsg
   * \copydetails ParseSyncResponseMsg(Msg* msg, int slice_idx);
   */
  void ParseResponseMsg(Msg* msg, int slice_idx);

 protected:
  int local_version_;
  //!< the ID of the first slice
  int slice_start_;
  int num_slices_;
  //!< offset and size of each slice
  vector<int> slice_offset_, slice_size_;

  //!< for debug checking
  vector<bool> pending_put_,pending_get_, pending_update_;
  int num_pending_requests_;

  shared_ptr<Blob<float>> data_;
  //! gradient, history gradient of this parameter
  Blob<float> grad_, history_;
  ParamProto proto_;
};

/**
 * ParamEntry is used for aggregating gradients of Params shared by workers from
 * the same group.
 *
 * For each worker group, every unique Param object has a ParamEntry object.
 * Param objects sharing the same values are associated with the same
 * ParamEntry.
 */
class ParamEntry{
 public:
  ParamEntry();
  ParamEntry(int total, Param* p);
  /**
   * Associate the counter to a Param object.
   *
   * @param p
   * @param local 1 if it is used by workers in this procs, 0 otherwise
   */
  void AddParam(bool local, Param* p);
  int num_update, next_version;
  int num_local; //!< # local workers using the shared parameter
  int num_total; //!< # total workers using the shared parameter
  //!< Shares are deleted by neuralnet's destructor
  vector<Param*> shares;
};

inline int ParamTrgt(int param_id, int slice_id) {
  return (param_id << 16) | slice_id;
}

inline int ParamID(int param_trgt) {
  return param_trgt >> 16;
}

inline int SliceID(int param_trgt) {
  static int mask = (1 << 16) -1;
  return param_trgt & mask;
}
}  // namespace singa

#endif  // INCLUDE_UTILS_PARAM_H_
