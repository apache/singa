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

#ifndef SINGA_UTILS_SINGLETON_H_
#define SINGA_UTILS_SINGLETON_H_

/// Thread-safe implementation for C++11 according to
//  http://stackoverflow.com/questions/2576022/efficient-thread-safe-singleton-in-c
template <typename T>
class Singleton {
 public:
  static T* Instance() {
    static T data_;
    return &data_;
  }
};

/// Thread Specific Singleton
/// Each thread will have its own data_ storage.
/*
template<typename T>
class TSingleton {
 public:
  static T* Instance() {
    static thread_local T data_;  // thread_local is not available in some
                                  // compilers
    return &data_;
  }
};
*/

#endif  // SINGA_UTILS_SINGLETON_H_
