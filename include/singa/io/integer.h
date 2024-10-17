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

#ifndef INTEGER_H_
#define INTEGER_H_

#include <cstdint>

namespace singa {
static bool isNetworkOrder() {
  int test = 1;
  return (1 != *(uint8_t*)&test);
}

template <typename T>
static inline T byteSwap(const T& v) {
  int size = sizeof(v);
  T ret;
  uint8_t* dest = reinterpret_cast<uint8_t*>(&ret);
  uint8_t* src = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&v));
  for (int i = 0; i < size; ++i) {
    dest[i] = src[size - i - 1];
  }
  return ret;
}

template <typename T>
static inline T hton(const T& v) {
  return isNetworkOrder() ? v : byteSwap(v);
}

template <typename T>
static inline T ntoh(const T& v) {
  return hton(v);
}

static inline int appendInteger(char* buf) { return 0; }
static inline int readInteger(char* buf) { return 0; }

template <typename Type, typename... Types>
static int appendInteger(char* buf, Type value, Types... values) {
  *(Type*)buf = hton(value);
  return sizeof(Type) + appendInteger(buf + sizeof(Type), values...);
}

template <typename Type, typename... Types>
static int readInteger(char* buf, Type& value, Types&... values) {
  value = ntoh(*(Type*)buf);
  return sizeof(Type) + readInteger(buf + sizeof(Type), values...);
}

}  // namespace singa
#endif
