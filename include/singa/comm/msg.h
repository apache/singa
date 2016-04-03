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

#ifndef SINGA_COMM_MSG_H_
#define SINGA_COMM_MSG_H_

#include <utility>

// TODO(wangwei): make it a compiler argument
// #define USE_ZMQ

#include <vector>
#ifdef USE_ZMQ
#include <czmq.h>
#endif

namespace singa {
/**
 * Wrapper to generate message address
 * @param grp worker/server group id
 * @param id_or_proc worker/server id or procs id
 * @param type msg type
 */
inline int Addr(int grp, int id_or_proc, int type) {
  return (grp << 16) | (id_or_proc << 8) | type;
}

/**
 * Parse group id from addr.
 *
 * @return group id
 */
inline int AddrGrp(int addr) {
  return addr >> 16;
}

/**
 * Parse worker/server id from addr.
 *
 * @return id
 */
inline int AddrID(int addr) {
  static const int mask = (1 << 8) - 1;
  return (addr >> 8) & mask;
}

/**
 * Parse worker/server procs from addr.
 *
 * @return procs id
 */
inline int AddrProc(int addr) {
  return AddrID(addr);
}

/**
 * Parse msg type from addr
 * @return msg type
 */
inline int AddrType(int addr) {
  static const int mask = (1 << 8) -1;
  return addr & mask;
}

/**
 * Msg used to transfer Param info (gradient or value), feature blob, etc.
 * between workers, stubs and servers.
 *
 * Each msg has a source addr and dest addr identified by a unique integer.
 * It is also associated with a target field (value and version) for ease of
 * getting some meta info (e.g., parameter id) from the msg.
 *
 * Other data is added into the message as frames.
 */
class Msg {
 public:
  ~Msg();
  Msg();
  /**
   * Construct the msg providing source and destination addr.
   */
  Msg(int src, int dst);
  /**
   * Copy constructor.
   */
  Msg(const Msg& msg);
  /**
   * Swap the src/dst addr
   */
  void SwapAddr();
  /**
   * Add a frame (a chunck of bytes) into the message
   */
  void AddFrame(const void* addr, int nBytes);
  /**
   * @return num of bytes of the current frame.
   */
  int FrameSize();
  /**
   * @return the pointer to the current frame data.
   */
  void* FrameData();
  /**
   * @return the data of the current frame as c string
   */
  char* FrameStr();
  /**
   * Move the cursor to the first frame.
   */
  void FirstFrame();
  /**
   * Move the cursor to the last frame.
   */
  void LastFrame();
  /**
   * Move the cursor to the next frame
   * @return true if the next frame is not NULL; otherwise false
   */
  bool NextFrame();
  /**
   *  Add a 'format' frame to the msg (like CZMQ's zsock_send).
   *
   *  The format is a string that defines the type of each field.
   *  The format can contain any of these characters, each corresponding to
   *  one or two arguments:
   *  i = int (signed)
   *  1 = uint8_t
   *  2 = uint16_t
   *  4 = uint32_t
   *  8 = uint64_t
   *  p = void * (sends the pointer value, only meaningful over inproc)
   *  s = char**
   *
   *  Returns size of the added content.
   */
  int AddFormatFrame(const char *format, ...);
  /**
   *  Parse the current frame added using AddFormatFrame(const char*, ...).
   *
   *  The format is a string that defines the type of each field.
   *  The format can contain any of these characters, each corresponding to
   *  one or two arguments:
   *  i = int (signed)
   *  1 = uint8_t
   *  2 = uint16_t
   *  4 = uint32_t
   *  8 = uint64_t
   *  p = void * (sends the pointer value, only meaningful over inproc)
   *  s = char**
   *
   *  Returns size of the parsed content.
   */
  int ParseFormatFrame(const char* format, ...);

#ifdef USE_ZMQ
  void ParseFromZmsg(zmsg_t* msg);
  zmsg_t* DumpToZmsg();
#endif

  /**
   * @return msg size in terms of bytes, ignore meta info.
   */
  int size() const;
  /**
   * Set source addr.
   * @param addr unique identify one worker/server/stub in the current job
   */
  inline void set_src(int addr) { src_ = addr; }
  /**
   * @return source addr.
   */
  inline int src() const { return src_; }
  /**
   * Set destination addr.
   * @param addr unique identify one worker/server/stub in the current job
   */
  inline void set_dst(int addr) { dst_ = addr; }
  /**
   * @return dst addr.
   */
  inline int dst() const { return dst_; }
  /**
   * Set msg type, e.g., kPut, kGet, kUpdate, kRequest
   */
  inline void set_type(int type) { type_ = type; }
  /**
   * @return msg type.
   */
  inline int type() const { return type_; }
  /**
   * Set msg target.
   *
   * One msg has a target to identify some entity in worker/server/stub.
   * The target is associated with a version, e.g., Param version.
   */
  inline void set_trgt(int val, int version) {
    trgt_val_ = val;
    trgt_version_ = version;
  }
  inline int trgt_val() const { return trgt_val_; }
  inline int trgt_version() const { return trgt_version_; }

 protected:
  int src_ = 0;
  int dst_ = 0;
  int type_ = 0;
  int trgt_val_ = 0;
  int trgt_version_ = 0;
#ifdef USE_ZMQ
  zmsg_t* msg_ = nullptr;
  zframe_t *frame_ = nullptr;
#else
  std::vector<std::pair<void*, int>> frames_;
  unsigned idx_ = 0;
#endif
};

inline void DeleteMsg(Msg** msg) {
  delete *msg;
  *msg = nullptr;
}

}  // namespace singa

#endif  // SINGA_COMM_MSG_H_
