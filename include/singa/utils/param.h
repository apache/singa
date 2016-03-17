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

#ifndef SINGA_UTILS_PARAM_H_
#define SINGA_UTILS_PARAM_H_

#include <memory>
#include <string>
#include <vector>

#include "singa/comm/msg.h"
#include "singa/proto/job.pb.h"
#include "singa/utils/blob.h"

namespace singa {
using std::vector;
/**
 * Base parameter generator which intializes parameter values.
 */
class ParamGenerator {
 public:
  static ParamGenerator* Create(const ParamGenProto& proto);

  virtual ~ParamGenerator() {}

  virtual void Init(const ParamGenProto& proto) { proto_ = proto; }
  virtual void Fill(Blob<float>* data);

 protected:
  ParamGenProto proto_;
};

class GaussianGen : public ParamGenerator {
 public:
  void  Fill(Blob<float>* data) override;
};

class GaussianSqrtFanInGen : public GaussianGen {
 public:
  void  Fill(Blob<float>* data) override;
};

class UniformGen : public ParamGenerator {
 public:
  void  Fill(Blob<float>* data) override;
};

class UniformSqrtFanInGen : public UniformGen {
 public:
  void Fill(Blob<float>* data) override;
};

class UniformSqrtFanInOutGen : public UniformGen {
 public:
  void Fill(Blob<float>* data) override;
};

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
class Param {
 public:
  /**
   * Create an instance of (sub) Param class based on the type from the
   * configuration.
   *
   * @param[in] conf configuration
   * @param a pointer to an instance
   */
  static Param* Create(const ParamProto& conf);

  /**
   * Try to slice the Param objects (from a neural net) into a given number of
   * servers (groups) evenly. This is to achieve load-balance among servers.
   *
   * It does not change the Param objects, but just computes the length of each
   * slice.
   *
   * @param num number of servers (groups) for maintaining the Param objects.
   * @param params all Param objects from a neural net.
   * @return the length of each slice.
   */
  static const vector<int> ComputeSlices(int num, const vector<Param*>& params);
  /**
   * It computes the length of each slice and slices the Param objects by adding
   * the slicing information into every Param object.
   *
   * @copydetails ComputeSlices()
   */
  static void SliceParams(int num, const vector<Param*>& params);

  Param() {}
  virtual ~Param() {}
  void Init(const ParamProto& proto) { proto_ = proto; }
  /**
   * Setup param object
   *
   * @param conf param configuration, include learning rate multiplier etc.
   * @param shape one value per dimension
   */
  virtual void Setup(const std::vector<int>& shape);
  /*
   * Fill the values according to init method, e.g., gaussian distribution.
   *
   * @param version initial version
   */
  virtual void InitValues();
  virtual void InitValues(int version);
  /**
   * Share the data blob from other Param objects.
   *
   * @param other the Param object whose owner owns the data blob
   * @param cpu_only if true, share only cpu memory (used for training with
   * multi-gpu cards); else, share both cpu and gpu memory.
   */
  void ShareDataFrom(Param* other, bool cpu_only);
  /**
   * Share both data and grad from other param
   */
  void ShareFrom(Param* other);
  /**
   * Init param values from checkpoint blob.
   */
  void FromProto(const BlobProto& blob);
  void FromProto(const std::string str);
  /**
   * Dump param values to blob.
   */
  void ToProto(BlobProto* blob);
  /**
   * Add a slice
   *
   * @param slice_id
   * @param size num of floats for this slice
   */
  void AddSlice(int slice_id, int size);
  /**
   * Scale the learning rate when updating parameters in the Param object
   */
  inline float lr_scale() const { return proto_.lr_scale(); }
  /**
   * Scale the weight decay when updating parameters in the Param object
   */
  inline float wd_scale() const { return proto_.wd_scale(); }
  /**
   * Parameter name used for Param re-use in other model or sharing between
   * layers
   */
  inline const std::string& name() const { return proto_.name(); }
  inline void set_name(const std::string& name) { proto_.set_name(name); }
  /**
   * If it shares data from others, then owner is the id of that Param,
   * otherwise it is itself's id.
   */
  inline int owner() const { return proto_.owner(); }
  /**
   * ID start from 0 and ordered for all Param from the same neuralnet
   */
  inline int id() const { return proto_.id(); }
  /**
   * Set ID
   */
  inline void set_id(int id) {
    proto_.set_id(id);
    proto_.set_owner(id);
  }
  inline int version() const { return version_; }
  inline void set_version(int v) { version_ = v; }
  /**
   * @return the version of the Param when the last Update request was issued.
   */
  inline int last_version() const { return last_version_; }
  inline void set_last_version(int v) { last_version_ = v; }

  /**
   * @return the sharing Param name which is configured by users in conf file.
   */
  inline const std::string& share_from() const { return proto_.share_from(); }
   /**
    * @return num of parameters in this Param obj.
    */
  inline const std::vector<int>& shape() const { return data_.shape(); }
  inline int size() const { return data_.count(); }
  inline const Blob<float>& data() const { return data_; }
  inline Blob<float>* mutable_data() { return &data_; }
  inline const Blob<float> &grad() const { return grad_; }
  inline Blob<float> *mutable_grad() { return &grad_; }
  inline float* mutable_cpu_data() { return data_.mutable_cpu_data(); }
  inline float* mutable_cpu_grad() { return grad_.mutable_cpu_data(); }
  inline float* mutable_cpu_history() { return history_.mutable_cpu_data(); }
  inline float* mutable_cpu_update() { return update_.mutable_cpu_data(); }
  /**
   * @return slice start ID
   */
  inline int slice_start() const { return slice_start_; }
  inline int num_slices() const { return num_slices_; }

  /**
   * Below are message/request related functions.
   * The basic communication workflows are as follow:
   *------------------------------------------------------------------------
   *         |Put         |Get           |Update           |Sync
   *------------------------------------------------------------------------
   * Generate|(stub)      |(stub)        |(stub)           |(server)
   * Message |GenPutMsg   |GenGetMsg     |GenUpdateMsg     |GenSyncMsg
   *------------------------------------------------------------------------
   * Handle  |(server)    |(server)      |(server)         |(server)
   * Message |HandlePutMsg|HandleGetMsg  |ParseUpdateMsg   |HandleSyncMsg
   *         |            |              |GenUpdateResMsg  |
   *------------------------------------------------------------------------
   * Handle  |            |(stub)        |(stub)           |(server)
   * Response|            |ParseGetResMsg|ParseUpdateResMsg|ParseSyncResMsg
   *------------------------------------------------------------------------
   */

  /**
   * Generate the message for a put request, i.e., put parameters to a server
   *
   * This function is called at worker/stub side.
   * @param copy decides whether to copy the parameter values from the server.
   * @param slice_idx index of the slice from which the message is generated.
   * @return generated message without setting src, dst, target fields.
   */
  virtual Msg* GenPutMsg(bool copy, int slice_idx);
  /**
   * Generate the message for a get request, i.e., get parameters from a server
   * \copydetails GenPutMsg(bool, int);
   */
  virtual Msg* GenGetMsg(bool copy, int slice_idx);
  /**
   * Generate the message for a update request, i.e., pass info to server for
   * parameter update.
   * \copydetails GenPutMsg(bool, int);
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
   * Server handling function for put request.
   *
   * @param msg request
   * @param reserve if true reserve the msg space for the calling function;
   * otherwise the msg should be freed inside the function.
   * @return resposne message
   */
  virtual Msg* HandlePutMsg(Msg** msg, bool reserve);
  /**
   * Server handling function for put request.
   *
   * \copydetails HandleGetMsg(Msg**, bool reserve)
   */
  virtual Msg* HandleGetMsg(Msg** msg, bool reserve);
  /**
   * Server parse update requests.
   * \copydetails GenUpdateResponseMsgs(const std::vector<Msg*>& msgs);
   */
  virtual void ParseUpdateMsgs(const std::vector<Msg*>& msgs);
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
  virtual const std::vector<Msg*>
    GenUpdateResponseMsgs(std::vector<Msg*>* msgs, bool reserve);
  /**
   * Server handling function for synchronization message
   *
   * \copydetails HandleGetMsg(Msg**, bool reserve)
   */
  virtual Msg* HandleSyncMsg(Msg** msg, bool reserve);
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
  //!< param version updated by the Update/Sync/Get response
  //!< only the owner param is initialized.
  int version_ = -1;
  //!< param version before last Update/Sync/Get request, set from version_
  int last_version_ = -1;
  //!< the global ID of the first slice
  int slice_start_ = 0;
  //!< total num of slices for this Parm obj
  int num_slices_ = 0;
  // offset and size of each slice
  std::vector<int> slice_offset_;
  std::vector<int> slice_size_;
  // for debug. Put request has no feedback, we do not track its pending status
  std::vector<bool> pending_get_;
  std::vector<bool> pending_update_;
  int num_pending_requests_ = 0;
  // data, gradient, history gradient of this parameter
  Blob<float> data_, grad_, history_, update_;
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
class ParamEntry {
 public:
  ParamEntry() {}
  ParamEntry(int total, Param* p);
  /**
   * Associate the counter to a Param object.
   *
   * @param p
   * @param local 1 if it is used by workers in this procs, 0 otherwise
   */
  void AddParam(bool local, Param* p);
  int next_version = -1;  // next_version & num_update are directly used by stub
  int num_update = 0;
  int num_local = 0;  //!< # local workers using the shared parameter
  int num_total = 0;  //!< # total workers using the shared parameter
  //!< Shares are deleted by neuralnet's destructor
  std::vector<Param*> shares;
};

inline int ParamTrgt(int param_id, int slice_id) {
  return (param_id << 16) | slice_id;
}

inline int ParamID(int param_trgt) {
  return param_trgt >> 16;
}

inline int SliceID(int param_trgt) {
  static const int mask = (1 << 16) -1;
  return param_trgt & mask;
}

}  // namespace singa

#endif  // SINGA_UTILS_PARAM_H_
